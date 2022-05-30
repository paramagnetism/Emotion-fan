#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from PIL import Image
from torchvision import transforms
import cv2
from basic_code import load, networks
#put model on cuda
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#label remark
rectify_label = {0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6}
#set pretrained model
class Predictor():
    def __init__(self,net_dir,face_detector_dir):
        self.face_detector=cv2.CascadeClassifier(face_detector_dir)
        _parameterDir = './pretrain_model/Resnet18_FER+_pytorch.pth.tar'
        structure = networks.resnet18_at(at_type=0)
        self.model = load.model_parameters(structure, _parameterDir)
        net=torch.load(net_dir)
        #model=model.to(DEVICE)
        self.model.load_state_dict(net['state_dict'])
        self.model.eval()
        self.transform=transforms.Compose([transforms.ToTensor()])
        
    def evaluate(self,vedio_pth,batch_size):
        '''
        batch_size shoule smaller than 64
        '''
        cap = cv2.VideoCapture(vedio_pth)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        videoWriter = cv2.VideoWriter('v2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), round(fps/24), size)
        batch=[]
        batch_count=0
        count=0
        scores=[]
        preds=[]
        while(cap.isOpened()): 
            _, img = cap.read()
            faces = self.face_detector.detectMultiScale(
               img,
               scaleFactor = 1.1,
               minNeighbors = 5,
               minSize = (10,10),
               flags = cv2.CASCADE_SCALE_IMAGE
            )
            #如果能检测到人脸
            if len(faces):
                [x,y,w,h]=faces[0]
                crop = cv2.resize(img[y:y+h,x:x+w,:],(224,224))
                cv22pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = self.transform(cv22pil)
                batch.append(tensor)
                batch_count+=1
                #print(count)
                #如果到了batch数，就打包去预测
                if batch_count == batch_size:
                    input_var = torch.stack(batch, dim=0).to(DEVICE)
                    with torch.no_grad():
                        f, alphas=self.model(input_var, phrase = 'eval')
                        weight_sourcefc = f.mul(alphas)
                        pred_score = self.model(vm=weight_sourcefc, phrase='eval', AT_level='pred')
                    score, pred = pred_score.topk(1, 1, True, True)
                    batch=[]
                    batch_count=0
                    for i in range(len(pred)):
                        preds.append(rectify_label[int(pred[i])])
                        scores.append(float(score[i].cpu()))
                    #cv2.imwrite("pic/"+ str(count)+".png",crop)
                #cv2.imshow('crop',crop)
                if preds:
                    cv2.putText(img,preds[-1]+ ": " + str(round(scores[-1],3)),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            count+=1
            videoWriter.write(img)
            cv2.imshow('image', img) 
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            k = cv2.waitKey(5)
            if (k & 0xff == ord('q')): 
                break
        cap.release() 
        videoWriter.release()
        cv2.destroyAllWindows()
        return scores, preds
        
if __name__ =="__main__":
    net_dir='model/self-attention_4_95.4545'
    face_detector_dir=r"model/haarcascade_frontalface_default.xml"
    vedio_pth="emotion.webm"
    predictor=Predictor(net_dir=net_dir, face_detector_dir=face_detector_dir)
    scores,preds=predictor.evaluate(vedio_pth,24)