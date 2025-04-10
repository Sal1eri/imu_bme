# -*- encoding: utf-8 -*-
# here put the import lib
import os.path

import torch
from torch import nn
from torchvision import models
import numpy as np
from torchvision.models import VGG16_Weights, ResNet34_Weights

pretrained_model = models.vgg16(weights=VGG16_Weights.DEFAULT)  # 用于 FCN32x FCN16x FCN8x
pretrained_net = models.resnet34(weights=ResNet34_Weights.DEFAULT)  # 用于 FCN8s


def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    双线性卷积核，用于反卷积
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()

        self.feature = pretrained_model.features

        self.conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.upsample32x = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_kernel(int(m.in_channels), int(m.out_channels), m.kernel_size[0]))

    def forward(self, x):
        x = self.feature(x)  # 1/32
        x = self.conv(x)
        x = self.upsample32x(x)
        return x


class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()

        self.feature_1 = nn.Sequential(*list(pretrained_model.features.children())[:24])
        self.feature_2 = nn.Sequential(*list(pretrained_model.features.children())[24:])

        self.conv_1 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)

        self.upsample2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1,
                                             output_padding=1, dilation=1)
        self.upsample16x = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))

    def forward(self, x):
        x1 = self.feature_1(x)
        x2 = self.feature_2(x1)

        x1 = self.conv_1(x1)
        x2 = self.conv_2(x2)
        x2 = self.upsample2x(x2)
        x2 += x1

        x2 = self.upsample16x(x2)
        return x2


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        self.feature_1 = nn.Sequential(*list(pretrained_model.features.children())[:17])
        self.feature_2 = nn.Sequential(*list(pretrained_model.features.children())[17:24])
        self.feature_3 = nn.Sequential(*list(pretrained_model.features.children())[24:])

        self.conv_1 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_3 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)

        self.upsample2x_1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1,
                                               output_padding=1, dilation=1)
        self.upsample2x_2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1,
                                               output_padding=1, dilation=1)
        self.upsample8x = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])

    def forward(self, x):
        x1 = self.feature_1(x)
        x2 = self.feature_2(x1)
        x3 = self.feature_3(x2)

        x2 = self.conv_1(x2)
        x3 = self.conv_3(x3)
        x3 = self.upsample2x_1(x3)
        x3 += x2

        x1 = self.conv_2(x1)
        x3 = self.upsample2x_2(x3)
        x3 += x1

        x3 = self.upsample8x(x3)
        return x3


class FCN8x(nn.Module):
    def __init__(self, num_classes):
        super(FCN8x, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])  # 第一段
        self.stage2 = list(pretrained_net.children())[-4]  # 第二段
        self.stage3 = list(pretrained_net.children())[-3]  # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def loadIFExist(self, model_path):

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_list = os.listdir('./model_result')

        model_pth = os.path.basename(model_path)

        if model_pth in model_list:
            self.load_state_dict(torch.load(model_path))
            print("the latest model has been load")

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/8
        x = self.stage2(x)
        s2 = x  # 1/16
        x = self.stage3(x)
        s3 = x  # 1/32
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s2)
        return s


if __name__ == "__main__":
    from torchsummary import summary

    fcn = FCN8x(2)
    fcn.cuda()
    summary(fcn,(3,224,224))
    # pretrained_model.cuda(1)
    # summary(pretrained_model,(3,128,128))
