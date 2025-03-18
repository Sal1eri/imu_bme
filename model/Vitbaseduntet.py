import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel


class ViT_UNet(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", num_classes=1):
        super(ViT_UNet, self).__init__()
        # 加载预训练的 ViT 模型
        self.vit = ViTModel.from_pretrained(vit_model_name)

        # 解码器部分（UNet 的上采样模块）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # ViT 提取特征
        vit_features = self.vit(x).last_hidden_state

        # 将 ViT 输出的特征序列转换为图像特征图
        vit_features = vit_features.permute(0, 2, 1).view(-1, 768, 14, 14)

        # 解码器部分
        x = self.decoder(vit_features)
        return x