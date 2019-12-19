import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from dataset_challenge import furdataset
import torch.nn as nn
import torchvision.models as models

from tensorboardX import SummaryWriter

# from se_resnet import se_resnet_101

writer_train = SummaryWriter("resNet101_fur_classification_run5/train")
writer_val = SummaryWriter("resNet101_fur_classification_run5/val")
wirter_all = SummaryWriter("resNet101_fur_classification_run5/all")

# 是否使用cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('CUDA: ', torch.cuda.is_available())

trainx_transforms = transforms.Compose([
    transforms.ToTensor(),   
    transforms.Normalize([0.5], [0.5])
])


def trainy_transforms(x):
    #img = transforms.ToTensor()(x)
    x = np.asarray(x)
    img = x.copy()
    img = torch.from_numpy(img)
    img = img.type(torch.LongTensor)
    return img


def test_model(model,criterion,dataload):
    model = model.eval()
    model = model.to(device)
    total_loss = 0
    total_acc = 0
    dt_size = len(dataload.dataset)
    for x,y in dataload:
        inputs = x.to(device)
        labels = y.to(device)

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)

        _,pred = outputs.max(1)
        num_correct = (pred==labels).sum().item()
        train_acc = num_correct/x.shape[0]

        total_loss = total_loss + loss.item()
        total_acc = total_acc + train_acc
    print("ave_test_acc:{},ave_test_loss:{}".format(total_acc/dt_size,total_loss/dt_size))

    return total_acc/dt_size,total_loss/dt_size


# 循环读取文件夹进行学习
def train_cir():
    #model = se_resnet_101().to(device)
    # 调用模型
    model = models.resnet101(pretrained=True)
    # 提取fc层中固定的参数（这一层的输入节点数目
    fc_features = model.fc.in_features
    # 修改类别为5，（直接对类的属性进行修改）
    model.fc = nn.Linear(fc_features, 5)
    model = model.to(device)

    batch_size = 8
    num_epochs = 500

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    root_train = "leather/train"
    root_val = "leather/val"
    train_image = os.path.join(root_train, "image")
    train_label = os.path.join(root_train, "label.txt")
    val_image = os.path.join(root_val, 'image')
    val_label = os.path.join(root_val, 'label.txt')

    train_iter = 0

    fur_dataset = furdataset(train_image, train_label, transform=trainx_transforms,target_transform=trainy_transforms)
    dataloaders = DataLoader(fur_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        train_iter += 1
        dt_size = len(dataloaders.dataset)
        epoch_loss = 0
        tot_acc = 0
        step = 0

        model = model.train()

        for x, y in dataloaders:
            step += 1
                
            inputs = x.to(device)
            labels = y.to(device)
                
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            
            # lab = labels.cpu().detach().numpy()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            # 计算分类的准确率
            _, pred = outputs.max(1)
            num_correct = (pred == labels).sum().item()
            train_acc = num_correct/x.shape[0]
            tot_acc += train_acc
            print("%d/%d,train_loss:%f,train_acc:%f" % (step, (dt_size - 1) // dataloaders.batch_size + 1,
                                                        loss.item(), train_acc))

        print("epoch %d ave_loss:%f,ave_acc:%f" % (epoch+1, epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),
                                                   tot_acc/((dt_size - 1) / dataloaders.batch_size + 1)))

        writer_train.add_scalar("train_loss", epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1), train_iter)
        writer_train.add_scalar("train_acc", tot_acc/((dt_size - 1) / dataloaders.batch_size + 1), train_iter)

        fur_dataset_test = furdataset(val_image, val_label, transform=trainx_transforms,
                                      target_transform=trainy_transforms)
        dataloaders_test = DataLoader(fur_dataset_test, batch_size=1)
        criterion = torch.nn.CrossEntropyLoss()
        model = model.eval()
        acc, loss = test_model(model, criterion, dataloaders_test)

        writer_val.add_scalar("val_loss", loss, train_iter)
        writer_val.add_scalar("val_acc", acc, train_iter)

        wirter_all.add_scalars("loss", {'train_loss': epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),
                                       'val_loss': loss}, train_iter)
        wirter_all.add_scalars("acc", {'train_acc': tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),
                                      'val_acc': acc}, train_iter)

        torch.save(model.state_dict(), '/resnet101_fur_classification__batch8_epoch{}model.pth'.format(epoch+1))
