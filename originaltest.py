#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import torch
from torchvision import transforms
from PIL import Image
from basic_code import load, networks
#put model on cuda
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#label remark
rectify_label = {0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6}
at_type='self-attention'
#set pretrained model
transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
_parameterDir = './pretrain_model/Resnet18_FER+_pytorch.pth.tar'
structure = networks.resnet18_at(at_type=1)
model = load.model_parameters(structure, _parameterDir)
net=torch.load('model/self-attention_4_95.4545')
#model=model.to(DEVICE)
model.load_state_dict(net['state_dict'])
model.eval()
#all fear
#img1='./data/face/ck_face/S054/001/S054_001_00000001.png'
img1='119.png'
img_first = Image.open(img1).convert("RGB")
#img2='./data/face/ck_face/S054/001/S054_001_00000007.png'
img2='215.png'
img_second = Image.open(img2).convert("RGB")
#img3='./data/face/ck_face/S054/001/S054_001_00000015.png'
img3='23.png'
img_third = Image.open(img3).convert("RGB")

img4=cv2.imread('23.png')
'''
import numpy as np

pil2cv2 = cv2.cvtColor(np.asarray(img_third), cv2.COLOR_RGB2BGR)
cv22pil = Image.fromarray(cv2.cvtColor(pil2cv2, cv2.COLOR_BGR2RGB))
cv22pil.show()
cv2.imshow("test",pil2cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()


dst_norm=np.empty(dst.shape,dtype=np.float32)
print(dst_norm.shape)
 
 
cv2.normalize(img,dst_norm,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#归一化
 
print(dst_norm)
'''
img4=cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)
cv2.normalize(img4,img4,alpha=0.0,beta=1.0,norm_type=cv2.NORM_MINMAX)
img_forth=torch.from_numpy(img4).permute(2,0,1)

img_first = transform(img_first)
img_second = transform(img_second)
img_third = transform(img_third)
input_var = torch.stack([img_first,img_second ,img_third,img_first,img_forth], dim=0).to(DEVICE)#其实是训练的时候拼接了三个图片做视频
with torch.no_grad():
    f, alphas=model(input_var, phrase = 'eval')
    weight_sourcefc = f.mul(alphas)     
    pred_score = model(vm=weight_sourcefc, phrase='eval', AT_level='pred')
_, pred = pred_score.topk(1, 1, True, True)
for i in range(len(pred)):
    print(rectify_label[int(pred[i])])