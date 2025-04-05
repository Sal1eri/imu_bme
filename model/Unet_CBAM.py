from collections import OrderedDict
import torch
import torch.nn as nn
import os
from torchviz import make_dot

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class UNet_CBAM(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=16):
        super(UNet_CBAM, self).__init__()

        features = init_features
        self.encoder1 = UNet_CBAM._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_CBAM._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_CBAM._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_CBAM._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_CBAM._block(features * 8, features * 16, name="bottleneck")

        # 添加CBAM模块
        self.cbam1 = CBAM(features)
        self.cbam2 = CBAM(features * 2)
        self.cbam3 = CBAM(features * 4)
        self.cbam4 = CBAM(features * 8)
        self.cbam_bottleneck = CBAM(features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_CBAM._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_CBAM._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_CBAM._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_CBAM._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc1 = self.cbam1(enc1)
        
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.cbam2(enc2)
        
        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.cbam3(enc3)
        
        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.cbam4(enc4)

        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.cbam_bottleneck(bottleneck)

        # 解码器部分
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))

    def save(self):
        name = "./saved_model/Unet_CBAM.pth"
        torch.save(self.state_dict(), name)
        print("the latest model has been saved")

    def loadIFExist(self):
        fileList = os.listdir('model_result')
        print(fileList)
        if "best_model_UNet_CBAM.mdl" in fileList:
            name = "./model_result/best_model_UNet_CBAM.mdl"
            self.load_state_dict(torch.load(name))
            print("the latest model has been loaded")

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

if __name__ == '__main__':
    from torchsummary import summary

    # 创建模型
    net = UNet_CBAM(3, 2)
    net.cuda()
    
    # 打印模型结构摘要
    summary(net, (3, 224, 224))
    
    # 创建示例输入
    x = torch.randn(1, 3, 224, 224).cuda()
    
    # 前向传播
    y = net(x)
    
    # 创建计算图
    dot = make_dot(y, params=dict(net.named_parameters()))
    
    # 保存为PDF文件
    dot.render("model_structure", format="pdf", cleanup=True)
    print("模型结构图已保存为 model_structure.pdf") 