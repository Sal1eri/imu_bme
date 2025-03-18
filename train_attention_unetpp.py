import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

from model.AttentionUNetPlusPlus import AttentionUNetPlusPlus
from utils.datasets import SegData
from utils.eval_tool import label_accuracy_score
from utils.transform import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize
from losses import SurfaceLoss

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train AttentionUNetPlusPlus')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--input-height', type=int, default=512, help='input height')
    parser.add_argument('--input-width', type=int, default=512, help='input width')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-interval', type=int, default=5, help='save model every n epochs')
    parser.add_argument('--train-image-path', type=str, default='data/training/images', help='path to training images')
    parser.add_argument('--train-mask-path', type=str, default='data/training/segmentations', help='path to training masks')
    parser.add_argument('--val-image-path', type=str, default='data/validation/images', help='path to validation images')
    parser.add_argument('--val-mask-path', type=str, default='data/validation/segmentations', help='path to validation masks')
    return parser.parse_args()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

def process_target(target):
    """处理标签，确保值在[0,1]范围内"""
    target = target.float()
    target = (target > 128).float()  # 二值化
    return target

def train_one_epoch(model, train_loader, criterion, dice_loss, surface_loss, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    total_mean_iu = 0
    total_fwavacc = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        target = process_target(target)  # 处理标签
        optimizer.zero_grad()
        
        outputs = model(data)
        
        # Deep supervision loss
        loss = 0
        weights = [0.5, 0.75, 0.875, 1.0]  # 权重递增
        for output, weight in zip(outputs, weights):
            # 组合损失：交叉熵和Dice损失
            ce_loss = criterion(output, target.squeeze().long())
            d_loss = dice_loss(output[:, 1:], target)
            loss += weight * (0.5 * ce_loss + 0.5 * d_loss)
        
        loss.backward()
        optimizer.step()
        
        # 计算评估指标（使用最后一个输出）
        pred = outputs[-1].argmax(dim=1)  # 使用argmax获取类别索引
        
        # 确保预测和标签是整数类型
        pred_np = pred.cpu().numpy().astype(np.int64)
        target_np = target.squeeze().cpu().numpy().astype(np.int64)
        
        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(target_np, pred_np, 2)
        
        total_loss += loss.item()
        total_acc += acc
        total_mean_iu += mean_iu
        total_fwavacc += fwavacc
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}',
            'IoU': f'{mean_iu:.4f}'
        })
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'acc': total_acc / len(train_loader),
        'mean_iu': total_mean_iu / len(train_loader),
        'fwavacc': total_fwavacc / len(train_loader)
    }
    return metrics

def validate(model, val_loader, criterion, dice_loss, surface_loss, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_mean_iu = 0
    total_fwavacc = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = process_target(target)  # 处理标签
            output = model(data)
            
            # 在验证时只使用最后一个输出
            if isinstance(output, list):
                output = output[-1]
            
            # 组合损失
            ce_loss = criterion(output, target.squeeze().long())
            d_loss = dice_loss(output[:, 1:], target)
            loss = 0.5 * ce_loss + 0.5 * d_loss
            
            # 计算评估指标
            pred = output.argmax(dim=1)  # 使用argmax获取类别索引
            
            # 确保预测和标签是整数类型
            pred_np = pred.cpu().numpy().astype(np.int64)
            target_np = target.squeeze().cpu().numpy().astype(np.int64)
            
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(target_np, pred_np, 2)
            
            total_loss += loss.item()
            total_acc += acc
            total_mean_iu += mean_iu
            total_fwavacc += fwavacc
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'acc': total_acc / len(val_loader),
        'mean_iu': total_mean_iu / len(val_loader),
        'fwavacc': total_fwavacc / len(val_loader)
    }
    return metrics

def main():
    args = get_args()
    device = torch.device(args.device)
    
    # 创建保存目录
    save_dir = './model_result'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./training_plots', exist_ok=True)
    
    # 数据预处理
    train_transform = Compose([
        Resize((args.input_height, args.input_width)),
        RandomHorizontalFlip(0.5),
        ToTensor()  # 移除Normalize，因为我们需要保持掩码的原始值
    ])
    
    val_transform = Compose([
        Resize((args.input_height, args.input_width)),
        ToTensor()  # 移除Normalize，因为我们需要保持掩码的原始值
    ])
    
    # 加载数据
    train_data = SegData(
        image_path=args.train_image_path,
        mask_path=args.train_mask_path,
        data_transforms=train_transform
    )
    
    val_data = SegData(
        image_path=args.val_image_path,
        mask_path=args.val_mask_path,
        data_transforms=val_transform
    )
    
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = AttentionUNetPlusPlus(in_channels=3, num_classes=2)
    model.deep_supervision = True  # 确保深度监督开启
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    surface_loss = SurfaceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # 训练日志
    best_miou = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_mean_iu': [], 'train_fwavacc': [],
        'val_loss': [], 'val_acc': [], 'val_mean_iu': [], 'val_fwavacc': []
    }
    
    print(f"开始训练，共{args.epochs}个epoch...")
    print(f"训练数据集大小: {len(train_data)}")
    print(f"验证数据集大小: {len(val_data)}")
    print(f"使用设备: {device}")
    print(f"学习率: {args.lr}")
    print(f"批次大小: {args.batch_size}")
    print(f"输入图像大小: {args.input_height}x{args.input_width}")
    
    for epoch in range(args.epochs):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, dice_loss, surface_loss, optimizer, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, dice_loss, surface_loss, device)
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 保存指标
        for k in train_metrics:
            history[f'train_{k}'].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])
        
        # 打印当前epoch的结果
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, "
              f"IoU: {train_metrics['mean_iu']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
              f"IoU: {val_metrics['mean_iu']:.4f}")
        
        # 保存最佳模型
        if val_metrics['mean_iu'] > best_miou:
            best_miou = val_metrics['mean_iu']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'history': history
            }, os.path.join(save_dir, 'best_attention_unetpp.pth'))
            print(f"保存最佳模型，当前最佳mIoU: {best_miou:.4f}")
        
        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'history': history
            }, os.path.join(save_dir, f'attention_unetpp_epoch_{epoch+1}.pth'))
        
        # 绘制训练曲线
        try:
            from utils.visualization import plot_training_curves
            plot_training_curves(history, './training_plots')
        except ImportError:
            print("可视化模块不可用，跳过绘制训练曲线")

if __name__ == '__main__':
    main() 