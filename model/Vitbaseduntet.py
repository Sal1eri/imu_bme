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
                nn.Sequential(
                    nn.Conv2d(self.encoder_channels[i], self.decoder_channels[i], kernel_size=1),
                    nn.BatchNorm2d(self.decoder_channels[i]),
                    nn.ReLU(inplace=True)
                )
            )

    def get_vit_parameters(self):
        """
        获取ViT编码器的参数
        Returns:
            list: ViT编码器的参数列表
        """
        return [p for p in self.vit.parameters() if p.requires_grad]
    
    def get_decoder_parameters(self):
        """
        获取解码器的参数
        Returns:
            list: 解码器的参数列表
        """
        decoder_params = []
        # 添加解码器各层的参数
        decoder_params.extend(self.decoder_blocks[0].parameters())
        decoder_params.extend(self.decoder_blocks[1].parameters())
        decoder_params.extend(self.decoder_blocks[2].parameters())
        decoder_params.extend(self.decoder_blocks[3].parameters())
        decoder_params.extend(self.final_conv.parameters())
        return decoder_params
    
    def unfreeze_vit(self, num_layers=None):
        """
        解冻ViT的指定层
        Args:
            num_layers: 要解冻的层数，None表示解冻所有层
        """
        # 首先冻结所有层
        for param in self.vit.parameters():
            param.requires_grad = False
            
        if num_layers is None:
            # 解冻所有层
            for param in self.vit.parameters():
                param.requires_grad = True
        else:
            # 解冻最后num_layers层
            total_layers = len(self.vit.encoder.layer)
            start_layer = max(0, total_layers - num_layers)
            
            # 解冻指定层的参数
            for i in range(start_layer, total_layers):
                for param in self.vit.encoder.layer[i].parameters():
                    param.requires_grad = True
            
            # 解冻patch embedding和position embedding
            for param in self.vit.embeddings.parameters():
                param.requires_grad = True

    def forward(self, x):
        # 保存输入尺寸
        input_size = x.size()[2:]
        
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
                # 使用正确的索引访问编码器特征
                skip = self.skip_connections[i](encoder_features[i])
                # 确保特征图大小匹配
                if x.size() != skip.size():
                    x = nn.functional.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
                x = x + skip
                
        # 最终输出
        x = self.final_conv(x)
        
        # 将输出调整到输入图像大小
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x
    
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
            
    def loadIFExist(self):
        """兼容原有训练代码的加载方法"""
        self.load()

