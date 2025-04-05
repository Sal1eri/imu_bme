from collections import OrderedDict

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import math
import torchvision.models as models


# 通道注意力特征模块(CAFM)
class CAFM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CAFM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 分组卷积，每个通道单独处理
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,  # 1x1卷积整合通道信息
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 实例批归一化模块(IBN)
class IBNorm(nn.Module):
    def __init__(self, channels, ratio=0.5):
        super(IBNorm, self).__init__()
        self.channels = channels
        self.ratio = ratio

        # 确定IN和BN的通道数
        self.in_channels = int(channels * ratio)
        self.bn_channels = channels - self.in_channels

        # 实例归一化用于处理域差异
        self.in_norm = nn.InstanceNorm2d(self.in_channels, affine=True)
        # 批归一化用于捕获判别性特征
        self.bn_norm = nn.BatchNorm2d(self.bn_channels)

    def forward(self, x):
        if self.ratio == 0.0:
            return self.bn_norm(x)
        elif self.ratio == 1.0:
            return self.in_norm(x)
        else:
            # 将特征通道分为两部分
            in_x, bn_x = torch.split(x, [self.in_channels, self.bn_channels], dim=1)
            # 分别应用IN和BN
            in_x = self.in_norm(in_x)
            bn_x = self.bn_norm(bn_x)
            # 重新合并
            return torch.cat([in_x, bn_x], dim=1)


# 高效通道注意力模块(ECA)
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 自适应地计算核大小
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 特征图的形状：[batch, channels, height, width]
        b, c, h, w = x.size()

        # 全局平均池化: [batch, channels, 1, 1]
        y = self.avg_pool(x)

        # 重新整形以适应1D卷积: [batch, 1, channels]
        y = y.squeeze(-1).transpose(-1, -2)

        # 通过1D卷积学习通道间的相互关系: [batch, 1, channels]
        y = self.conv(y)

        # 重新整形为原始形状: [batch, channels, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)

        # 通过sigmoid门控并与原始特征相乘
        y = self.sigmoid(y)
        return x * y


# 空间注意力模块(用于CBAM)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        scale = self.sigmoid(y)
        return x * scale


# 卷积块注意力模块(CBAM)
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = CAFM(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# 空洞卷积模块
class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, padding=2):
        super(DilatedConv, self).__init__()
        self.dilated_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 空洞空间金字塔池化模块(ASPP)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 全局池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 1x1卷积分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 多个不同膨胀率的空洞卷积分支
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 合并分支的1x1卷积
        self.conv_final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()

        # 全局池化分支
        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=True)

        # 不同膨胀率的空洞卷积分支
        conv1_features = self.conv1(x)
        conv2_features = self.conv2(x)
        conv3_features = self.conv3(x)
        conv4_features = self.conv4(x)

        # 合并所有分支
        combined_features = torch.cat([
            global_features,
            conv1_features,
            conv2_features,
            conv3_features,
            conv4_features
        ], dim=1)

        # 通过1x1卷积整合特征
        output = self.conv_final(combined_features)

        return output


# 带残差连接的卷积块 - 使用IBNorm
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_depthwise=True, name="", ibn_ratio=0.5):
        super(ResBlock, self).__init__()

        self.use_depthwise = use_depthwise
        self.needs_projection = (in_channels != out_channels)

        # 如果输入输出通道数不同，需要1x1卷积进行调整
        if self.needs_projection:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            # 对于捷径也使用IBNorm
            self.proj_norm = IBNorm(out_channels, ratio=ibn_ratio)

        # 主路径第一个卷积
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # 使用IBNorm替代BatchNorm
        self.norm1 = IBNorm(out_channels, ratio=ibn_ratio)
        self.relu1 = nn.ReLU(inplace=True)

        # 主路径第二个卷积
        if use_depthwise:
            self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # 使用IBNorm替代BatchNorm
        self.norm2 = IBNorm(out_channels, ratio=ibn_ratio)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # 残差连接
        if self.needs_projection:
            identity = self.projection(x)
            identity = self.proj_norm(identity)

        # 残差相加
        out += identity
        out = self.relu2(out)

        return out


# ResNet-18骨干网络，带IBNorm
class ResNet18IBN(nn.Module):
    def __init__(self, pretrained=True, ibn_ratio=0.5):
        super(ResNet18IBN, self).__init__()

        # 加载预训练的ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # 初始层
        self.conv1 = resnet.conv1
        self.ibn1 = IBNorm(64, ratio=ibn_ratio)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 残差层
        self.layer1 = resnet.layer1  # 64通道，1/4分辨率
        self.layer2 = resnet.layer2  # 128通道，1/8分辨率
        self.layer3 = resnet.layer3  # 256通道，1/16分辨率
        self.layer4 = resnet.layer4  # 512通道，1/32分辨率

    def forward(self, x):
        # 保存各层特征用于跳跃连接
        features = []

        # 初始层
        x = self.conv1(x)
        x = self.ibn1(x)
        x0 = self.relu(x)  # 64通道，1/2分辨率
        features.append(x0)

        x = self.maxpool(x0)

        # 残差层
        x1 = self.layer1(x)  # 64通道，1/4分辨率
        features.append(x1)

        x2 = self.layer2(x1)  # 128通道，1/8分辨率
        features.append(x2)

        x3 = self.layer3(x2)  # 256通道，1/16分辨率
        features.append(x3)

        x4 = self.layer4(x3)  # 512通道，1/32分辨率
        features.append(x4)

        return features


class UNetImprove(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=16, use_depthwise=True, ibn_ratio=0.5,
                 use_resnet=True):
        super(UNetImprove, self).__init__()

        features = init_features
        self.use_depthwise = use_depthwise
        self.ibn_ratio = ibn_ratio
        self.use_resnet = use_resnet

        # 添加原始尺寸处理路径 - 参考Qnet设计
        if use_depthwise:
            self.conv_original_size0 = nn.Sequential(
                DepthwiseSeparableConv(in_channels, 64, kernel_size=3, padding=1),
                IBNorm(64, ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )
            self.conv_original_size1 = nn.Sequential(
                DepthwiseSeparableConv(64, 64, kernel_size=3, padding=1),
                IBNorm(64, ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_original_size0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
                IBNorm(64, ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )
            self.conv_original_size1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                IBNorm(64, ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )

        # 通用上采样模块
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if use_resnet:
            # 使用ResNet18作为编码器骨干
            self.encoder = ResNet18IBN(pretrained=True, ibn_ratio=ibn_ratio)
            # ResNet18编码器的特征维度[64, 64, 128, 256, 512]
            encoder_dims = [64, 64, 128, 256, 512]

            # 瓶颈层 - 调整通道维度
            self.bottleneck = nn.Sequential(
                DepthwiseSeparableConv(encoder_dims[4], encoder_dims[4], kernel_size=3, padding=1),
                IBNorm(encoder_dims[4], ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )

            # 添加CBAM和ASPP模块
            self.cbam_bottleneck = CBAM(encoder_dims[4])
            self.aspp = ASPP(encoder_dims[4], encoder_dims[4])

            # 1x1卷积调整通道
            self.layer0_1x1 = nn.Sequential(
                nn.Conv2d(encoder_dims[0], encoder_dims[0], kernel_size=1, bias=False),
                IBNorm(encoder_dims[0], ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )
            self.layer1_1x1 = nn.Sequential(
                nn.Conv2d(encoder_dims[1], encoder_dims[1], kernel_size=1, bias=False),
                IBNorm(encoder_dims[1], ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )
            self.layer2_1x1 = nn.Sequential(
                nn.Conv2d(encoder_dims[2], encoder_dims[2], kernel_size=1, bias=False),
                IBNorm(encoder_dims[2], ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )
            self.layer3_1x1 = nn.Sequential(
                nn.Conv2d(encoder_dims[3], encoder_dims[3], kernel_size=1, bias=False),
                IBNorm(encoder_dims[3], ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )
            self.layer4_1x1 = nn.Sequential(
                nn.Conv2d(encoder_dims[4], encoder_dims[4], kernel_size=1, bias=False),
                IBNorm(encoder_dims[4], ratio=ibn_ratio),
                nn.ReLU(inplace=True)
            )

            # 解码器卷积块
            self.conv_up3 = ResBlock(encoder_dims[3] + encoder_dims[4], encoder_dims[4], use_depthwise, name="up3",
                                     ibn_ratio=ibn_ratio)
            self.conv_up2 = ResBlock(encoder_dims[2] + encoder_dims[4], encoder_dims[3], use_depthwise, name="up2",
                                     ibn_ratio=ibn_ratio)
            self.conv_up1 = ResBlock(encoder_dims[1] + encoder_dims[3], encoder_dims[3], use_depthwise, name="up1",
                                     ibn_ratio=ibn_ratio)
            self.conv_up0 = ResBlock(encoder_dims[0] + encoder_dims[3], encoder_dims[2], use_depthwise, name="up0",
                                     ibn_ratio=ibn_ratio)

            # 最后的融合块，处理原始尺寸特征
            self.conv_original_size2 = ResBlock(64 + encoder_dims[2], 64, use_depthwise, name="orig_fusion",
                                                ibn_ratio=ibn_ratio)

            # 添加ECA注意力模块
            self.eca0 = ECA(encoder_dims[0])
            self.eca1 = ECA(encoder_dims[1])
            self.eca2 = ECA(encoder_dims[2])
            self.eca3 = ECA(encoder_dims[3])
            self.eca4 = ECA(encoder_dims[4])

            # 最终输出层
            if use_depthwise:
                self.conv = nn.Sequential(
                    DepthwiseSeparableConv(64, 64, kernel_size=3, padding=1),
                    nn.Conv2d(64, out_channels, kernel_size=1)
                )
            else:
                self.conv = nn.Conv2d(
                    in_channels=64, out_channels=out_channels, kernel_size=1
                )

        else:
            # 原始编码器 - 使用ResBlock
            self.encoder1 = ResBlock(in_channels, features, use_depthwise, name="enc1", ibn_ratio=ibn_ratio)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.encoder2 = ResBlock(features, features * 2, use_depthwise, name="enc2", ibn_ratio=ibn_ratio)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.encoder3 = ResBlock(features * 2, features * 4, use_depthwise, name="enc3", ibn_ratio=ibn_ratio)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.encoder4 = ResBlock(features * 4, features * 8, use_depthwise, name="enc4", ibn_ratio=ibn_ratio)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

            # 瓶颈 - 使用ResBlock
            self.bottleneck = ResBlock(features * 8, features * 16, use_depthwise, name="bottleneck",
                                       ibn_ratio=ibn_ratio)

            # 解码器
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder4 = ResBlock((features * 8) * 2, features * 8, use_depthwise, name="dec4", ibn_ratio=ibn_ratio)

            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = ResBlock((features * 4) * 2, features * 4, use_depthwise, name="dec3", ibn_ratio=ibn_ratio)

            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = ResBlock((features * 2) * 2, features * 2, use_depthwise, name="dec2", ibn_ratio=ibn_ratio)

            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = ResBlock(features * 2, features, use_depthwise, name="dec1", ibn_ratio=ibn_ratio)

            # 添加额外的上采样和原始尺寸融合层
            self.conv_original_size2 = ResBlock(features + 64, 64, use_depthwise, name="orig_fusion",
                                                ibn_ratio=ibn_ratio)

            # 最终输出层
            if use_depthwise:
                self.conv = nn.Sequential(
                    DepthwiseSeparableConv(64, 64, kernel_size=3, padding=1),
                    nn.Conv2d(64, out_channels, kernel_size=1)
                )
            else:
                self.conv = nn.Conv2d(
                    in_channels=64, out_channels=out_channels, kernel_size=1
                )

            # 添加CBAM和ASPP模块实例
            self.cbam_bottleneck = CBAM(features * 16)
            self.aspp = ASPP(features * 16, features * 16)

            # 添加ECA注意力模块到各个编码器之间
            self.eca1 = ECA(features)
            self.eca2 = ECA(features * 2)
            self.eca3 = ECA(features * 4)
            self.eca4 = ECA(features * 8)

            # 为跳跃连接添加CBAM模块
            self.cbam_skip1 = CBAM(features)
            self.cbam_skip2 = CBAM(features * 2)
            self.cbam_skip3 = CBAM(features * 4)
            self.cbam_skip4 = CBAM(features * 8)

    def forward(self, x):
        # 首先处理原始尺寸特征 - 参考Qnet设计
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        if self.use_resnet:
            # 使用ResNet18编码器
            features = self.encoder(x)

            # 提取各层特征并应用ECA和1x1卷积
            layer0 = features[0]  # 64通道，1/2分辨率
            layer0 = self.eca0(layer0)

            layer1 = features[1]  # 64通道，1/4分辨率
            layer1 = self.eca1(layer1)

            layer2 = features[2]  # 128通道，1/8分辨率
            layer2 = self.eca2(layer2)

            layer3 = features[3]  # 256通道，1/16分辨率
            layer3 = self.eca3(layer3)

            layer4 = features[4]  # 512通道，1/32分辨率
            layer4 = self.eca4(layer4)

            # 瓶颈处理
            layer4 = self.layer4_1x1(layer4)
            bottleneck = self.bottleneck(layer4)
            bottleneck = self.cbam_bottleneck(bottleneck)
            bottleneck = self.aspp(bottleneck)

            # 解码器上采样过程 - 参考Qnet的上采样和融合方式
            x = self.upsample(bottleneck)  # 1/16分辨率
            layer3 = self.layer3_1x1(layer3)
            x = torch.cat([x, layer3], dim=1)
            x = self.conv_up3(x)

            x = self.upsample(x)  # 1/8分辨率
            layer2 = self.layer2_1x1(layer2)
            x = torch.cat([x, layer2], dim=1)
            x = self.conv_up2(x)

            x = self.upsample(x)  # 1/4分辨率
            layer1 = self.layer1_1x1(layer1)
            x = torch.cat([x, layer1], dim=1)
            x = self.conv_up1(x)

            x = self.upsample(x)  # 1/2分辨率
            layer0 = self.layer0_1x1(layer0)
            x = torch.cat([x, layer0], dim=1)
            x = self.conv_up0(x)

            # 最终上采样到原始尺寸并融合原始特征
            x = self.upsample(x)  # 1/1分辨率(原始尺寸)
            x = torch.cat([x, x_original], dim=1)
            x = self.conv_original_size2(x)

        else:
            # 使用原始编码器
            enc1 = self.encoder1(x)
            enc1_att = self.eca1(enc1)

            enc2 = self.encoder2(self.pool1(enc1_att))
            enc2_att = self.eca2(enc2)

            enc3 = self.encoder3(self.pool2(enc2_att))
            enc3_att = self.eca3(enc3)

            enc4 = self.encoder4(self.pool3(enc3_att))
            enc4_att = self.eca4(enc4)

            bottleneck = self.bottleneck(self.pool4(enc4_att))
            bottleneck = self.cbam_bottleneck(bottleneck)
            bottleneck = self.aspp(bottleneck)

            dec4 = self.upconv4(bottleneck)
            skip4 = self.cbam_skip4(enc4_att)
            dec4 = torch.cat((dec4, skip4), dim=1)
            dec4 = self.decoder4(dec4)

            dec3 = self.upconv3(dec4)
            skip3 = self.cbam_skip3(enc3_att)
            dec3 = torch.cat((dec3, skip3), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            skip2 = self.cbam_skip2(enc2_att)
            dec2 = torch.cat((dec2, skip2), dim=1)
            dec2 = self.decoder2(dec2)

            dec1 = self.upconv1(dec2)
            skip1 = self.cbam_skip1(enc1_att)
            dec1 = torch.cat((dec1, skip1), dim=1)
            dec1 = self.decoder1(dec1)

            # 额外上采样处理和与原始尺寸特征融合
            dec1 = self.upsample(dec1)  # 确保恢复到原始尺寸
            dec1 = torch.cat([dec1, x_original], dim=1)
            dec1 = self.conv_original_size2(dec1)

        return self.conv(x if self.use_resnet else dec1)

    def save(self):
        name = "./saved_model/UNetImprove.pth"
        torch.save(self.state_dict(), name)
        print("UNetImprove模型已保存")

    def loadIFExist(self):
        fileList = os.listdir('model_result')
        print(fileList)
        if "best_model_UNetImprove.mdl" in fileList:
            model_path = "./model_result/best_model_UNetImprove.mdl"
            self.load_state_dict(torch.load(model_path))
            print("UNetImprove模型已加载")
        else:
            print("未找到UNetImprove模型，使用初始化参数")


if __name__ == '__main__':
    from torchsummary import summary

    # 使用ResNet18作为编码器
    net = UNetImprove(3, 2, use_depthwise=False, ibn_ratio=0.5, use_resnet=True)
    net.cuda()
    summary(net, (3, 224, 224))

