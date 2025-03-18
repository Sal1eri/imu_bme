import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel


class ViT_UNet(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", num_classes=2):
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

        # 去掉 CLS token，只保留 patch embeddings
        vit_features = vit_features[:, 1:, :]  # 形状为 [batch_size, 196, 768]

        # 将特征重新排列为图像形状
        vit_features = vit_features.view(-1, 768, 14, 14)  # 形状为 [batch_size, 768, 14, 14]

        # 解码器部分
        x = self.decoder(vit_features)
        return x