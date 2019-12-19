import torch.utils.data as data
import PIL.Image as Image
import os


def load2dict(root_label):
    f = open(root_label,'r')
    result_dict = {}
    for line in f:
        v = line.strip().split('-')
        result_dict[v[0]] = v[1]
    f.close()
    return result_dict


#    root_image = "leather/train/image"    root_label = "leather/train/label.txt"
def make_dataset(root_image,root_label):
    imgs=[]
    n = len(os.listdir(root_image))
    ls = os.listdir(root_image)
    label_dict = load2dict(root_label)
    for i in range(n):
        img=os.path.join(root_image,ls[i])
        label = label_dict[ls[i]]
        imgs.append((img,label))
    return imgs


class furdataset(data.Dataset):
    def __init__(self, root_image,root_label, transform=None, target_transform=None):
        imgs = make_dataset(root_image,root_label)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y = self.imgs[index]
        img_x = Image.open(x_path)       
        img_x = img_x.convert('RGB')#位深度24转换为8，转为灰度图

        label = int(y)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img_x, label

    def __len__(self):
        return len(self.imgs)






