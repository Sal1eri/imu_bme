# -*- encoding: utf-8 -*-
# here put the import lib
import pandas as pd
import numpy as np
from utils.DataLoade import CustomDataset
from torch.utils.data import DataLoader
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
from model.DeepLab import DeepLabV3
from model.swin_transformer_v2 import SegFormer
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from utils.eval_tool import label_accuracy_score
from utils.data_txt import image2csv
import argparse
from tqdm import tqdm
import time
from losses import SurfaceLoss
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#   引用u3+模型
from u3plus.UNet_3Plus import UNet_3Plus
from u3plus.UNet_3Plus import UNet_3Plus_DeepSup
from u3plus.Qnet import ResNetUNet
from u3plus.Ues50 import UesNet
from u3plus.U2plusRes50 import NestedUResnet,BasicBlock

#   引用parser
from CommandLine.train_parser import get_args_parser

#   引用psp
from model.PSPnet import PSPNet

from utils.datasets import SegData

from utils.transform import Resize,Compose,ToTensor,Normalize,RandomHorizontalFlip


# 引用loss

from model.Vitbaseduntet import ViT_UNet
from model.AttentionUNetPlusPlus import AttentionUNetPlusPlus


def boundary_loss(data, label):
    """
    计算边界损失
    Args:
        data: 模型输出的预测结果 [B, C, H, W]
        label: 真实标签 [B, H, W]
    Returns:
        loss: 边界损失值
    """
    from losses import SurfaceLoss, class2one_hot, one_hot2dist
    
    # 确保输入在正确的设备上
    device = data.device
    
    # 将预测结果转换为one-hot格式
    pred = data.argmax(dim=1)  # [B, H, W]
    pred_one_hot = class2one_hot(pred, 2)  # [B, 2, H, W]
    
    # 将标签转换为one-hot格式
    label_one_hot = class2one_hot(label, 2)  # [B, 2, H, W]
    
    # 计算距离图
    dist_maps = []
    for i in range(label_one_hot.size(0)):  # 对每个样本分别处理
        dist_map = one_hot2dist(label_one_hot[i].cpu().numpy())
        dist_maps.append(dist_map)
    dist_maps = torch.from_numpy(np.stack(dist_maps)).to(device)
    
    # 计算边界损失
    surface_loss = SurfaceLoss()
    loss = surface_loss(pred_one_hot, dist_maps, None)
    
    return loss

def load_model(args):
    if args.model == 'Unet':
        model_name = 'UNet'
        net = UNet(3, 2)
        net.loadIFExist()
        print("using UNet")
    elif args.model == "FCN":
        model_name = 'FCN8x'
        net = FCN8x(args.n_classes)
        print("using FCN")
    elif args.model == "Deeplab":
        model_name = 'DeepLabV3'
        net = DeepLabV3(args.n_classes)
        print("using DeeplabV3")
    elif args.model == 'Unet3+':
        model_name = 'Unet3+'
        net = UNet_3Plus()
        print("using UNet3+")
    elif args.model == 'Qnet':
        model_name = 'Qnet'
        net = ResNetUNet()
        print("using ResNetUNet")
    elif args.model == 'Uesnet50':
        model_name = 'Uesnet50'
        net = UesNet()
        print('using Uesnet50')
    elif args.model == 'Unet2+':
        model_name = 'URestnet++'
        net = NestedUResnet(block=BasicBlock,layers=[3,4,6,3],num_classes=2)
        print('using URestnet2+')
    elif args.model == 'ViT':
        model_name = 'ViT'
        net = ViT_UNet(num_classes=2)
        print('using ViT')
    elif args.model == 'AttentionUNet+':
        model_name = 'AttentionUNet++'
        net = AttentionUNetPlusPlus()
        print('using AttentionUNet+')
    else:
        model_name = 'PSPnet'
        net = PSPNet(3)
        print("using PSPnet")


    return model_name, net

def plot_training_curves(history, save_dir, model_name):
    """
    Plot training and validation metrics
    Args:
        history: Dictionary containing all training metrics
        save_dir: Directory to save the plots
        model_name: Name of the model
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建画布
    plt.figure(figsize=(12, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # IoU曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_mean_iu'], label='Train IoU')
    plt.plot(history['val_mean_iu'], label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU Curves')
    plt.legend()
    plt.grid(True)
    
    # Dice系数曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Curves')
    plt.legend()
    plt.grid(True)
    
    # 像素准确率曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Accuracy')
    plt.title('Pixel Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(save_dir, f'{model_name}_results.png'))
    plt.close()
    
    print(f'指标曲线图已保存至 {os.path.join(save_dir, f"{model_name}_results.png")}')


def train(args, model_name, net):
    model_path = './model_result/best_model_{}.mdl'.format(model_name)
    result_path = './result_{}.txt'.format(model_name)
    # 获取模型保存路径的父目录
    parent_dir = os.path.dirname(model_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if os.path.exists(result_path):
        os.remove(result_path)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    surface_criterion = SurfaceLoss()
    
    # 设置优化器
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=1e-4)
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5, 
        verbose=True
    )
    
    # 设置混合损失权重
    alpha = 1  # 交叉熵损失权重
    beta = 0   # 边界损失权重
    
    best_score = 0.0
    start_time = time.time()
    
    # train_csv_dir = 'test.csv'
    # val_csv_dir = 'val.csv'
    #
    #
    # train_data = CustomDataset(train_csv_dir, args.input_height, args.input_width)
    # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #
    # val_data = CustomDataset(val_csv_dir, args.input_height, args.input_width)
    # val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

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


    # train_image_root = args.data_path + '/images/train'
    # train_lables_root = args.data_path + '/lables/train'

    #train_dataset = torchvision.datasets.ImageFolder

    train_dataset = SegData(image_path=os.path.join(args.data_path, 'training/images'),
                            mask_path=os.path.join(args.data_path, 'training/segmentations'),
                            data_transforms=train_transform)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    val_dataset = SegData(image_path=os.path.join(args.data_path, 'validation/images'),
                            mask_path=os.path.join(args.data_path, 'validation/segmentations'),
                            data_transforms=val_transform)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.cuda()
        criterion = criterion.cuda()
    epoch = args.epochs
    
    # 创建训练历史记录字典
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_acc_cls': [],
        'val_acc_cls': [],
        'train_mean_iu': [],
        'val_mean_iu': [],
        'train_fwavacc': [],
        'val_fwavacc': [],
        'learning_rates': [],
        'train_dice': [],
        'val_dice': [],
        'train_recall': [],
        'val_recall': []
    }
    
    # 创建图表保存目录
    plots_dir = os.path.join('training_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for e in range(epoch):
        net.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        
        with tqdm(total=len(train_dataloader), desc=f'{e + 1}/{epoch} epoch Train_Progress') as pb_train:
            for i, (batchdata, batchlabel) in enumerate(train_dataloader):
                if use_gpu:
                    batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()
                    batchlabel = (batchlabel / 255).to(device).long()
                
                # 前向传播
                output = net(batchdata)
                output = F.log_softmax(output, dim=1)
                
                # 计算混合损失
                ce_loss = criterion(output, batchlabel)
                bd_loss = boundary_loss(output, batchlabel)
                loss = alpha * ce_loss + beta * bd_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 记录损失和预测
                train_loss += loss.cpu().item() * batchlabel.size(0)
                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = batchlabel.data.cpu()
                label_true = torch.cat((label_true, real), dim=0)
                label_pred = torch.cat((label_pred, pred), dim=0)
                
                pb_train.update(1)
        
        # 计算训练指标
        train_loss /= len(train_dataloader.dataset)
        acc, acc_cls, mean_iu, fwavacc, iu, acc_cls_per_class, dice, mean_dice, recall, mean_recall = label_accuracy_score(
            label_true.numpy(), 
            label_pred.numpy(),
            args.n_classes
        )
        
        # 更新学习率
        scheduler.step(train_loss)
        
        print(f'epoch: {e + 1}')
        print(f'train_loss: {train_loss:.4f}')
        print(f'Overall Metrics:')
        print(f'  - Accuracy: {acc:.4f}')
        print(f'  - Mean Accuracy: {acc_cls:.4f}')
        print(f'  - Mean IoU: {mean_iu:.4f}')
        print(f'  - Weighted Accuracy: {fwavacc:.4f}')
        print(f'  - Mean Dice: {mean_dice:.4f}')
        print(f'  - Mean Recall: {mean_recall:.4f}')
        print(f'Per-class Metrics:')
        for i in range(args.n_classes):
            print(f'  Class {i}:')
            print(f'    - IoU: {iu[i]:.4f}')
            print(f'    - Accuracy: {acc_cls_per_class[i]:.4f}')
            print(f'    - Dice: {dice[i]:.4f}')
            print(f'    - Recall: {recall[i]:.4f}')
        print(f'Time for this epoch: {time.time() - epoch_start_time:.2f} seconds')
        
        # 保存结果
        with open(result_path, 'a') as f:
            f.write(f'\nepoch: {e + 1}\n')
            f.write(f'train_loss: {train_loss:.4f}\n')
            f.write('Overall Metrics:\n')
            f.write(f'  - Accuracy: {acc:.4f}\n')
            f.write(f'  - Mean Accuracy: {acc_cls:.4f}\n')
            f.write(f'  - Mean IoU: {mean_iu:.4f}\n')
            f.write(f'  - Weighted Accuracy: {fwavacc:.4f}\n')
            f.write(f'  - Mean Dice: {mean_dice:.4f}\n')
            f.write(f'  - Mean Recall: {mean_recall:.4f}\n')
            f.write('Per-class Metrics:\n')
            for i in range(args.n_classes):
                f.write(f'  Class {i}:\n')
                f.write(f'    - IoU: {iu[i]:.4f}\n')
                f.write(f'    - Accuracy: {acc_cls_per_class[i]:.4f}\n')
                f.write(f'    - Dice: {dice[i]:.4f}\n')
                f.write(f'    - Recall: {recall[i]:.4f}\n')
        
        # 记录训练指标
        history['train_loss'].append(train_loss)
        history['train_acc'].append(acc)
        history['train_acc_cls'].append(acc_cls)
        history['train_mean_iu'].append(mean_iu)
        history['train_fwavacc'].append(fwavacc)
        history['train_dice'].append(mean_dice)
        history['train_recall'].append(mean_recall)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 验证阶段
        net.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        
        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'{e + 1}/{epoch} epoch Val_Progress') as pb_val:
                for i, (batchdata, batchlabel) in enumerate(val_dataloader):
                    if use_gpu:
                        batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()
                        batchlabel = (batchlabel / 255).to(device).long()
                    
                    output = net(batchdata)
                    output = F.log_softmax(output, dim=1)
                    
                    # 计算验证损失
                    ce_loss = criterion(output, batchlabel)
                    bd_loss = boundary_loss(output, batchlabel)
                    loss = alpha * ce_loss + beta * bd_loss
                    
                    pred = output.argmax(dim=1).squeeze().data.cpu()
                    real = batchlabel.data.cpu()
                    
                    val_loss += loss.cpu().item() * batchlabel.size(0)
                    val_label_true = torch.cat((val_label_true, real), dim=0)
                    val_label_pred = torch.cat((val_label_pred, pred), dim=0)
                    
                    pb_val.update(1)
        
        # 计算验证指标
        val_loss /= len(val_dataloader.dataset)
        val_acc, val_acc_cls, val_mean_iu, val_fwavacc, val_iu, val_acc_cls_per_class, val_dice, val_mean_dice, val_recall, val_mean_recall = label_accuracy_score(
            val_label_true.numpy(),
            val_label_pred.numpy(),
            args.n_classes
        )
        
        print(f'val_loss: {val_loss:.4f}')
        print(f'Overall Metrics:')
        print(f'  - Accuracy: {val_acc:.4f}')
        print(f'  - Mean Accuracy: {val_acc_cls:.4f}')
        print(f'  - Mean IoU: {val_mean_iu:.4f}')
        print(f'  - Weighted Accuracy: {val_fwavacc:.4f}')
        print(f'  - Mean Dice: {val_mean_dice:.4f}')
        print(f'  - Mean Recall: {val_mean_recall:.4f}')
        print(f'Per-class Metrics:')
        for i in range(args.n_classes):
            print(f'  Class {i}:')
            print(f'    - IoU: {val_iu[i]:.4f}')
            print(f'    - Accuracy: {val_acc_cls_per_class[i]:.4f}')
            print(f'    - Dice: {val_dice[i]:.4f}')
            print(f'    - Recall: {val_recall[i]:.4f}')
        
        # 记录验证指标
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_acc_cls'].append(val_acc_cls)
        history['val_mean_iu'].append(val_mean_iu)
        history['val_fwavacc'].append(val_fwavacc)
        history['val_dice'].append(val_mean_dice)
        history['val_recall'].append(val_mean_recall)
        
        # 保存最佳模型
        score = (val_acc_cls + val_mean_iu + val_mean_dice + val_mean_recall) / 4
        if score > best_score:
            best_score = score
            torch.save(net.state_dict(), model_path)
            print(f'Best model saved with score: {best_score:.4f}')
    
    # 训练结束后，绘制最终图表
    plot_training_curves(history, plots_dir, model_name)
    
    # 保存训练历史到CSV文件
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(plots_dir, f'{model_name}_training_history.csv'), index=False)
    
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # args.model = 'AttentionUNet+'

    model_name, net = load_model(args)
    # print(args.n_classes)
    # print(args.init_lr)
    train(args,model_name,net)
