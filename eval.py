from tqdm import tqdm

from utils.DataLoade import CustomDataset
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
from model.FCN import FCN8x
from utils.eval_tool import label_accuracy_score
from model.DeepLab import DeepLabV3
import numpy as np

import time

from model.DeepLab import DeepLabV3
from u3plus.UNet_3Plus import UNet_3Plus
from model.PSPnet import PSPNet
from u3plus.Qnet import ResNetUNet

BATCH_SIZE = 16
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
GPU_ID = 0
NUM_CLASSES = 2


def main():
    val_csv_dir = 'val.csv'
    val_data = CustomDataset(val_csv_dir, INPUT_WIDTH, INPUT_HEIGHT)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # model = 'Unet3+'
    # net = UNet_3Plus()

    # model = 'PSPnet'
    # net = PSPNet(num_classes=NUM_CLASSES)
    #
    # model = 'FCN8x'
    # net = FCN8x(NUM_CLASSES)

    # model = 'DeepLabV3'
    # net = DeepLabV3(21)

    model = 'Qnet'
    net = ResNetUNet()

    model_path = './model_result/best_model_{}.mdl'.format(model)

    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.cuda()
    net.eval()

    use_gpu = torch.cuda.is_available()

    torch.cuda.set_device(GPU_ID)

    pbar = tqdm(total=len(val_dataloader))
    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.NLLLoss()

    val_loss = 0.0
    val_label_true = torch.LongTensor()
    val_label_pred = torch.LongTensor()

    with torch.no_grad():
        for i, (batchdata, batchlabel) in enumerate(val_dataloader):
            if use_gpu:
                batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()

            output = net(batchdata)
            # loss_soft_before = criterion(output, batchlabel)
            output = F.log_softmax(output, dim=1)
            start_time1 = time.perf_counter()
            loss = criterion(output, batchlabel)
            end_time1 = time.perf_counter()


            # loss1t=end_time1-start_time1
            # print("log soft ", loss1t)
            # start_time2 = time.perf_counter()
            # loss2 = criterion2(output,batchlabel)
            # end_time2 = time.perf_counter()
            # loss2t=end_time2-start_time2
            # ratio = loss1t/loss2t
            # print("NLLos",loss2t)
            # input = F.log_softmax(output, dim=1)
            # loss3 = criterion2(input, batchlabel)

            pred = output.argmax(dim=1).squeeze().data.cpu()
            real = batchlabel.data.cpu()
            val_loss += loss.cpu().item() * batchlabel.size(0)
            val_label_true = torch.cat((val_label_true, real), dim=0)
            val_label_pred = torch.cat((val_label_pred, pred), dim=0)

            pbar.update(1)

        val_loss /= len(val_data)
        val_acc, val_acc_cls, val_mean_iu, val_fwavacc, cls_iou, cls_acc, dice, mean_dice = label_accuracy_score(
            val_label_true.numpy(),
            val_label_pred.numpy(),
            NUM_CLASSES)
        #   val_acc
        #   val_acc_cls
        #   val_mean_iu
        #   val_fwavacc
        print()
        print(
            f'val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, '
            f'mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}, mean_dice: {mean_dice:.4f}')

    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'potted plant',
               'sheep', 'sofa', 'train', 'tv/monitor']
    print("==========Every IOU==========")
    for name, prob in zip(classes, cls_iou):
        print(name + " : " + str(prob))
    print("=============================")

    draw_img(classes, cls_iou)

def draw_img(classes,cls_iou):
    import matplotlib.pyplot as plt

    # 生成颜色列表，确保每个柱形颜色不同
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(classes)))

    # 绘制柱形图
    plt.figure(figsize=(10, 8))
    bars = plt.bar(range(len(classes)), cls_iou, color=colors)

    # 使用类别名称作为横坐标
    plt.xlabel('Classes')
    plt.ylabel('Intersection over Union (IoU)')
    plt.title('IoU for Each Class')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')  # 使用类别名称，并旋转使其更易读
    plt.ylim(0.0, 1.0)  # 设置y轴范围为0到1
    plt.grid(axis='y')  # 显示y轴的网格线

    # 添加数值标签
    for bar, iou in zip(bars, cls_iou):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{iou:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    # 保存图像到当前目录下的文件
    plt.savefig('./iou_bar_chart.png', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()
