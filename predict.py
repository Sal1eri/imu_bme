import pandas as pd
import numpy as np
from utils.DataLoade import CustomDataset
# from torch.utils.data import DataLoader
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
import torch
import os
from model.DeepLab import DeepLabV3
# from torch import nn,optim
# from torch.nn import functional as F
from utils.eval_tool import label_accuracy_score
import history

# model = 'UNet'
model = 'FCN8x'
# model = 'DeepLabV3'
GPU_ID = 0
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
BATCH_SIZE = 8
NUM_CLASSES = 21
LEARNING_RATE = 1e-3

model_path = './model_result/best_model_{}.mdl'.format(model)
# model_path='./model_result/best_model_{}_kaggle.mdl'.format(model)

torch.cuda.set_device(GPU_ID)
net = FCN8x(NUM_CLASSES)


# net = DeepLabV3(NUM_CLASSES)
# net = UNet(3,NUM_CLASSES)
# 加载网络进行测试


def evaluate(model):
    import random
    from utils.DataLoade import label2image, RandomCrop
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from PIL import Image
    # @qyk
    from tqdm import tqdm

    test_csv_dir = 'test.csv'
    testset = CustomDataset(test_csv_dir, INPUT_WIDTH, INPUT_HEIGHT)
    test_dataloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    # index = random.randint(0, len(testset) - 1)
    # index = [5,6]
    pbar = tqdm(total=BATCH_SIZE)
    for (val_image, val_label) in test_dataloader:
        # val_image, val_label = test_dataloader[1]
        net.cuda()
        out = net(val_image.cuda())  # [Batch_size, NUM_CLASSES, INPUT_HEIGHT, INPUT_WIDTH]
        pred = out.argmax(dim=1).squeeze().data.cpu().numpy()  # [Batch_size, INPUT_HEIGHT, INPUT_WIDTH]
        label = val_label.data.numpy()
        val_pred, val_label = label2image(NUM_CLASSES)(pred, label)

        for i in range(BATCH_SIZE):
            val_imag = val_image[i]
            val_pre = val_pred[i]
            val_labe = val_label[i]
            # 反归一化
            mean = [.485, .456, .406]
            std = [.229, .224, .225]
            x = val_imag
            for j in range(3):
                x[j] = x[j].mul(std[j]) + mean[j]
            img = x.mul(255).byte()
            img = img.numpy().transpose((1, 2, 0))  # 原图

            fig, ax = plt.subplots(1, 3, figsize=(30, 30))
            ax[0].imshow(img)
            ax[1].imshow(val_labe)
            ax[2].imshow(val_pre)
            # plt.show()
            plt.savefig('./pic_results/pic_{}_{}.png'.format(model, i))
            # 更新进度条
            pbar.update(1)
        break  # 只显示一个batch 否则会一直生成下去



if __name__ == "__main__":
    evaluate(model)

