import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import os
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PyramidPoolingModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in pool_sizes])
        self.bottleneck = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=3,
                                    padding=1)

    def _make_stage(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in
                  self.stages] + [x]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.pyramid_pooling = PyramidPoolingModule(in_channels=2048, pool_sizes=[1, 2, 3, 6], out_channels=512)
        self.final = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
    def loadIFExist(self, model_path):
        model_list = os.listdir('./model_result')

        model_pth = os.path.basename(model_path)

        if model_pth in model_list:
            self.load_state_dict(torch.load(model_path))
            print("the latest model has been load")

    def forward(self, x):
        size = (x.size()[2], x.size()[3])
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.pyramid_pooling(x)
        x = self.final(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    net = PSPNet(21)
    net.cuda()
    summary(net, (3, 128, 128))