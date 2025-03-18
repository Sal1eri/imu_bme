# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import time
import os
import numpy as np
from model.Vitbaseduntet import ViT_UNet
from utils.datasets import SegData
from utils.transform import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip
from utils.eval_tool import label_accuracy_score
from CommandLine.train_parser import get_args_parser

def calculate_metrics(pred, target, num_classes):
    """
    计算多个评估指标
    Args:
        pred: 预测标签
        target: 真实标签
        num_classes: 类别数
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))
    for t, p in zip(target, pred):
        confusion_matrix[t, p] += 1
    
    # 计算每个类别的IoU
    iou_list = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - intersection
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        iou_list.append(iou)
    
    # 计算平均IoU
    mean_iou = np.mean(iou_list)
    
    # 计算Dice系数
    dice_list = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        total = np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i])
        if total == 0:
            dice = 0
        else:
            dice = 2 * intersection / total
        dice_list.append(dice)
    mean_dice = np.mean(dice_list)
    
    # 计算加权准确率
    total_pixels = np.sum(confusion_matrix)
    weighted_acc = 0
    for i in range(num_classes):
        class_pixels = np.sum(confusion_matrix[i, :])
        if class_pixels > 0:
            class_acc = confusion_matrix[i, i] / class_pixels
            weighted_acc += class_acc * (class_pixels / total_pixels)
    
    return {
        'iou': iou_list,
        'mean_iou': mean_iou,
        'dice': dice_list,
        'mean_dice': mean_dice,
        'weighted_acc': weighted_acc
    }

def train_vit(args, num_layers=None, vit_lr_ratio=0.1):
    """
    训练ViT模型，支持微调
    Args:
        args: 训练参数
        num_layers: 要解冻的ViT层数，None表示解冻所有层
        vit_lr_ratio: ViT学习率与解码器学习率的比例
    """
    # 创建模型
    model = ViT_UNet(num_classes=2)
    
    # 设置保存路径
    model_path = f'./model_result/best_model_ViT_finetune_{num_layers}layers.mdl'
    result_path = f'./result_ViT_finetune_{num_layers}layers.txt'
    
    # 创建保存目录
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 如果结果文件存在则删除
    if os.path.exists(result_path):
        os.remove(result_path)
    
    # 解冻ViT的指定层
    model.unfreeze_vit(num_layers)
    
    # 获取ViT和解码器的参数
    vit_params = model.get_vit_parameters()
    decoder_params = model.get_decoder_parameters()
    
    # 创建优化器，为ViT和解码器设置不同的学习率
    optimizer = optim.Adam([
        {'params': vit_params, 'lr': args.init_lr * vit_lr_ratio},
        {'params': decoder_params, 'lr': args.init_lr}
    ], weight_decay=1e-4)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 数据预处理
    train_transform = Compose([
        Resize((args.input_height, args.input_width)),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = Compose([
        Resize((args.input_height, args.input_width)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据加载器
    train_dataset = SegData(
        image_path=os.path.join(args.data_path, 'training/images'),
        mask_path=os.path.join(args.data_path, 'training/segmentations'),
        data_transforms=train_transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataset = SegData(
        image_path=os.path.join(args.data_path, 'validation/images'),
        mask_path=os.path.join(args.data_path, 'validation/segmentations'),
        data_transforms=val_transform
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    
    # 训练循环
    best_score = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        
        with tqdm(total=len(train_dataloader), desc=f'{epoch + 1}/{args.epochs} epoch Train_Progress') as pb_train:
            for batchdata, batchlabel in train_dataloader:
                batchdata = batchdata.to(device)
                batchlabel = (batchlabel / 255).to(device).long()
                
                # 前向传播
                output = model(batchdata)
                output = F.log_softmax(output, dim=1)
                loss = criterion(output, batchlabel)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.vit.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 记录损失和预测
                train_loss += loss.item() * batchlabel.size(0)
                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = batchlabel.data.cpu()
                label_true = torch.cat((label_true, real), dim=0)
                label_pred = torch.cat((label_pred, pred), dim=0)
                
                pb_train.update(1)
        
        # 计算训练指标
        train_loss /= len(train_dataloader.dataset)
        train_metrics = calculate_metrics(
            label_pred.numpy(),
            label_true.numpy(),
            args.n_classes
        )
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        
        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'{epoch + 1}/{args.epochs} epoch Val_Progress') as pb_val:
                for batchdata, batchlabel in val_dataloader:
                    batchdata = batchdata.to(device)
                    batchlabel = (batchlabel / 255).to(device).long()
                    
                    output = model(batchdata)
                    output = F.log_softmax(output, dim=1)
                    loss = criterion(output, batchlabel)
                    
                    val_loss += loss.item() * batchlabel.size(0)
                    pred = output.argmax(dim=1).squeeze().data.cpu()
                    real = batchlabel.data.cpu()
                    val_label_true = torch.cat((val_label_true, real), dim=0)
                    val_label_pred = torch.cat((val_label_pred, pred), dim=0)
                    
                    pb_val.update(1)
        
        # 计算验证指标
        val_loss /= len(val_dataloader.dataset)
        val_metrics = calculate_metrics(
            val_label_pred.numpy(),
            val_label_true.numpy(),
            args.n_classes
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f'epoch: {epoch + 1}')
        print(f'Train - Loss: {train_loss:.4f}, Weighted Acc: {train_metrics["weighted_acc"]:.4f}, '
              f'mIoU: {train_metrics["mean_iou"]:.4f}, mDice: {train_metrics["mean_dice"]:.4f}')
        print(f'Val - Loss: {val_loss:.4f}, Weighted Acc: {val_metrics["weighted_acc"]:.4f}, '
              f'mIoU: {val_metrics["mean_iou"]:.4f}, mDice: {val_metrics["mean_dice"]:.4f}')
        
        # 保存结果
        with open(result_path, 'a') as f:
            f.write(f'\nepoch: {epoch + 1}\n')
            f.write(f'Train - Loss: {train_loss:.4f}, Weighted Acc: {train_metrics["weighted_acc"]:.4f}, '
                   f'mIoU: {train_metrics["mean_iou"]:.4f}, mDice: {train_metrics["mean_dice"]:.4f}\n')
            f.write(f'Val - Loss: {val_loss:.4f}, Weighted Acc: {val_metrics["weighted_acc"]:.4f}, '
                   f'mIoU: {val_metrics["mean_iou"]:.4f}, mDice: {val_metrics["mean_dice"]:.4f}\n')
            # 保存每个类别的IoU和Dice
            for i in range(args.n_classes):
                f.write(f'Class {i} - IoU: {train_metrics["iou"][i]:.4f}, Dice: {train_metrics["dice"][i]:.4f}\n')
        
        # 保存最佳模型（使用mIoU作为指标）
        if val_metrics['mean_iou'] > best_score:
            best_score = val_metrics['mean_iou']
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with mIoU: {best_score:.4f}')
    
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    
    # 可以尝试不同的微调策略
    # 1. 只微调最后4层
    train_vit(args, num_layers=4)
    
    # 2. 微调最后6层
    # train_vit(args, num_layers=6)
    
    # 3. 完全微调
    # train_vit(args, num_layers=None)
    
    # 4. 使用不同的学习率比例
    # train_vit(args, num_layers=4, vit_lr_ratio=0.05) 