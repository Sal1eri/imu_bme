# -*- encoding: utf-8 -*-
# here put the import lib
import pandas as pd
import numpy as np
from utils.DataLoade import CustomDataset
from torch.utils.data import DataLoader
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
from model.DeepLab import DeepLabV3
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from utils.eval_tool import label_accuracy_score
from utils.data_txt import image2csv
import argparse
from tqdm import tqdm
import time

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
        net = NestedUResnet(block=BasicBlock,layers=[3,4,6,3],num_classes=21)
        print('using URestnet2+')
    else:
        model_name = 'PSPnet'
        net = PSPNet(3)
        print("using PSPnet")


    return model_name, net


def train(args, model_name, net):
    model_path = './model_result/best_model_{}.mdl'.format(model_name)
    result_path = './result_{}.txt'.format(model_name)
    # 获取模型保存路径的父目录
    parent_dir = os.path.dirname(model_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if os.path.exists(result_path):
        os.remove(result_path)

    best_score = 0.0
    start_time = time.time()  # 开始训练的时间
    # 加载模型


    # 构建网络
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()


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
    for e in range(epoch):
        net.train()
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        #   train的进度条
        with tqdm(total=len(train_dataloader), desc=f'{e + 1}/{epoch} epoch Train_Progress') as pb_train:
            for i, (batchdata, batchlabel) in enumerate(train_dataloader):
                if use_gpu:
                    batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()
                    batchlabel = (batchlabel / 255).to(device).long()

                output = net(batchdata)
                output = F.log_softmax(output, dim=1)
                loss = criterion(output, batchlabel)

                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = batchlabel.data.cpu()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item() * batchlabel.size(0)
                label_true = torch.cat((label_true, real), dim=0)
                label_pred = torch.cat((label_pred, pred), dim=0)
                pb_train.update(1)

        train_loss /= len(train_dataloader.dataset)
        acc, acc_cls, mean_iu, fwavacc, _, _, _, _ = label_accuracy_score(label_true.numpy(), label_pred.numpy(),
                                                                          args.n_classes)

        print(
            f'epoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f}')
        print(f'Time for this epoch: {time.time() - epoch_start_time:.2f} seconds')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f}')

        net.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        with tqdm(total=len(val_dataloader), desc=f'{e + 1}/{epoch} epoch Val_Progress') as pb_val:
            with torch.no_grad():
                for i, (batchdata, batchlabel) in enumerate(val_dataloader):
                    if use_gpu:
                        batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()
                        batchlabel = (batchlabel / 255).to(device).long()

                    output = net(batchdata)
                    output = F.log_softmax(output, dim=1)
                    loss = criterion(output, batchlabel)

                    pred = output.argmax(dim=1).squeeze().data.cpu()
                    real = batchlabel.data.cpu()

                    val_loss += loss.cpu().item() * batchlabel.size(0)
                    val_label_true = torch.cat((val_label_true, real), dim=0)
                    val_label_pred = torch.cat((val_label_pred, pred), dim=0)

                    pb_val.update(1)

            val_loss /= len(val_dataloader.dataset)
            val_acc, val_acc_cls, val_mean_iu, val_fwavacc, _, _, _, _ = label_accuracy_score(val_label_true.numpy(),
                                                                                              val_label_pred.numpy(),
                                                                                              args.n_classes)

        print(
            f'epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}')

        score = (val_acc_cls + val_mean_iu) / 2
        if score > best_score:
            best_score = score
            torch.save(net.state_dict(), model_path)

    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    model_name, net = load_model(args)
    # print(args.n_classes)
    # print(args.init_lr)
    train(args,model_name,net)
