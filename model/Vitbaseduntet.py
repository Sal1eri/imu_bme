import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel
import os


class ViT_UNet(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", num_classes=2):
        super(ViT_UNet, self).__init__()
        # 加载预训练的 ViT 模型
        self.vit = ViTModel.from_pretrained(vit_model_name)
        
        # 冻结ViT参数
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 编码器特征维度
        self.encoder_channels = [768, 768, 768, 768]  # ViT base的特征维度
        
        # 解码器部分
        self.decoder_channels = [512, 256, 128, 64]
        
        # 解码器模块
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(self.decoder_channels)):
            in_channels = self.encoder_channels[i] if i == 0 else self.decoder_channels[i-1]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, self.decoder_channels[i], kernel_size=2, stride=2),
                    nn.BatchNorm2d(self.decoder_channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.decoder_channels[i], self.decoder_channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.decoder_channels[i]),
                    nn.ReLU(inplace=True)
                )
            )
            
        # 最终输出层
        self.final_conv = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)
        
        # 跳跃连接
        self.skip_connections = nn.ModuleList()
        for i in range(len(self.decoder_channels)):
            self.skip_connections.append(
                nn.Conv2d(self.encoder_channels[i], self.decoder_channels[i], kernel_size=1)
            )

    def forward(self, x):
        # ViT 提取特征
        vit_output = self.vit(x)
        features = vit_output.last_hidden_state
        
        # 去掉 CLS token，只保留 patch embeddings
        features = features[:, 1:, :]  # [batch_size, 196, 768]
        
        # 将特征重新排列为图像形状
        features = features.view(-1, 768, 14, 14)  # [batch_size, 768, 14, 14]
        
        # 存储编码器特征
        encoder_features = []
        current_feature = features
        
        # 编码器特征提取
        for i in range(len(self.encoder_channels)):
            if i > 0:
                current_feature = nn.MaxPool2d(2)(current_feature)
            encoder_features.append(current_feature)
            
        # 解码器部分
        x = encoder_features[-1]
        for i in range(len(self.decoder_blocks)):
            # 上采样
            x = self.decoder_blocks[i](x)
            
            # 跳跃连接
            if i < len(self.skip_connections):
                skip = self.skip_connections[i](encoder_features[-(i+2)])
                x = x + skip
                
        # 最终输出
        x = self.final_conv(x)
        return torch.sigmoid(x)
    
    def save(self, path="./model_result/best_model_ViT_UNet.mdl"):
        """保存模型"""
        torch.save(self.state_dict(), path)
        print(f"模型已保存到: {path}")
        
    def load(self, path="./model_result/best_model_ViT_UNet.mdl"):
        """加载模型"""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"模型已从 {path} 加载")
        else:
            print(f"未找到模型文件: {path}")
            
    def unfreeze_vit(self, num_layers=None):
        """解冻ViT的部分层"""
        if num_layers is None:
            for param in self.vit.parameters():
                param.requires_grad = True
        else:
            # 只解冻最后几层
            for name, param in self.vit.named_parameters():
                if any(f"layer.{i}" in name for i in range(12-num_layers, 12)):
                    param.requires_grad = True