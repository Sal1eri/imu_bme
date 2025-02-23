# -*- encoding: utf-8 -*-
# here put the import lib
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
import torch

random.seed(78)

class CustomDataset(Dataset):
    def __init__(self, data_root_csv, input_width, input_height, test=False):
        super(CustomDataset, self).__init__()
        self.data_root_csv = data_root_csv
        self.data_all = pd.read_csv(self.data_root_csv)
        self.image_list = list(self.data_all.iloc[:, 0])
        self.label_list = list(self.data_all.iloc[:, 1])
        self.width = input_width
        self.height = input_height

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert('RGB')
        label = Image.open(self.label_list[index]).convert('L')  # 以灰度模式打开标签图像

        img, label = self.train_transform(img, label, crop_size=(self.width, self.height))

        return img, label

    def train_transform(self, image, label, crop_size=(256, 256)):
        image, label = RandomCrop(crop_size)(image, label)
        tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        image = tfs(image)

        label = np.array(label, dtype=np.int64)
        label[label == 255] = 1  # 将像素值为 255 的区域标记为 1
        label = torch.from_numpy(label).long()
        return image, label


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, label):
        i, j, h, w = self.get_params(img, self.size)
        return img.crop((j, i, j + w, i + h)), label.crop((j, i, j + w, i + h))


class label2image():
    def __init__(self):
        self.colormap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

    def __call__(self, label_pred, label_true):
        pred = self.colormap[label_pred]
        # true = self.colormap[label_true]
        label_true[label_true != 0] = 1
        true = label_true
        return pred, true



if __name__ == "__main__":
    pass
    # DATA_ROOT = './data/'
    # traindata = CustomTrainDataset(DATA_ROOT,256,256)
    # traindataset = DataLoader(traindata,batch_size=2,shuffle=True,num_workers=0)

    # for i,batch in enumerate(traindataset):
    #     img,label = batch
    #     print(img,label)

    # l1 = Image.open('data/SegmentationClass/2007_000032.png').convert('RGB')

    # label = image2label()(l1)
    # print(label[150:160, 240:250])