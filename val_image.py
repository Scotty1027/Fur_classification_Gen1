import PIL.Image as Image
import os
import random
import numpy as np
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from resnet101 import resnet101
import time


img_h = 256
img_w = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainx_transforms = transforms.Compose([
    transforms.ToTensor(),   
    transforms.Normalize([0.5], [0.5])
])

def val_image(x_path):
    model = resnet101()
    #提取fc层中固定的参数  这一层的输入节点数目
    fc_features = model.fc.in_features
    #修改类别为5，（直接对类的属性进行修改）
    model.fc = nn.Linear(fc_features, 5)

    model = model.to(device)
    imgxs = []
    results = []

    model.load_state_dict(torch.load('/resnet101_fur_classification__batch8_epoch23model.pth', map_location='cuda:0'))

    model.eval()

    img = Image.open(x_path)       
    img = img.convert('RGB')
    img = np.asarray(img)
    X_height,X_width,_ = img.shape
    for i in range(20):
        random_width = random.randint(0, X_width - img_w - 1)
        random_height = random.randint(0, X_height - img_h - 1)
        src_roi = img[random_height: random_height + img_h, random_width: random_width + img_w,:]
        imgxs.append(src_roi)

    for j in range(len(imgxs)):
        img_x = imgxs[j]
        img_x = trainx_transforms(img_x)
        img_x = torch.unsqueeze(img_x,0)
        img_x = img_x.to(device)
        outputs = model(img_x)
        _,pred = outputs.max(1)
        results.append(pred.item())
    results = np.array(results)
    classification = np.argmax(np.bincount(results))
    return classification


start = time.time()
x_path = "0.bmp"
classification = val_image(x_path)
end = time.time()
print(classification)
print('time: ' + str(end-start) + 's')
