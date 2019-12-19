import torch
import numpy as np
import torch.nn as nn


def IoU(prediction, target):
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    delta = 1e-10
    IoU = ((prediction * target).sum() + delta) / (prediction.sum() + target.sum() - (prediction * target).sum() + delta)

    return IoU


def Dice(prediction, target):
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    delta = 1e-10
    return (2 * (prediction * target).sum() + delta) / (prediction.sum() + target.sum() + delta)


def IOU(input,target):
    batch,_,_ = input.shape

    acc = 0
    for i in range(batch):
        a = input[i]
        b = target[i]
        a[a>=0.5] = 1
        a[a<0.5] = 0
        iou = IoU(a,b)
        acc = acc + iou
    return acc/batch

