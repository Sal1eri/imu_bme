import numpy as np
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
import torch
import torchvision.transforms as transforms
import os
from model.DeepLab import DeepLabV3
from parameters import *
#   引用u3+模型
from u3plus.UNet_3Plus import UNet_3Plus
from u3plus.UNet_3Plus import UNet_3Plus_DeepSup
from u3plus.Qnet import ResNetUNet
from u3plus.Ues50 import UesNet
from u3plus.U2plusRes50 import NestedUResnet,BasicBlock
#   引用psp
from model.PSPnet import PSPNet


def evaluate(val_image_path, model_e):
    from utils.DataLoade import colormap
    import matplotlib.pyplot as plt
    from PIL import Image
    model_path = MODEL_path + 'best_model_{}.mdl'.format(model_e)
    if model_e == 'FCN8x':
        net = FCN8x(NUM_CLASSES)
    elif model_e == 'UNet':
        net = UNet(3, NUM_CLASSES)
    elif model_e == 'DeepLabV3':
        net = DeepLabV3(NUM_CLASSES)
    elif model_e == 'Unet3+':
        net = UNet_3Plus()
    elif model_e == 'Qnet':
        net = ResNetUNet()
    elif model_e == 'PSPnet':
        net = PSPNet(NUM_CLASSES)
    elif model_e == 'Uesnet50':
        net = UesNet()
    elif model_e == 'Unet2+':
        net = NestedUResnet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=21)
    else:
        net = FCN8x(NUM_CLASSES)

    net.eval()
    image_name = os.path.basename(val_image_path)
    image_name = image_name.replace(".jpg", ".png")
    val_image = Image.open(val_image_path)
    tfs = transforms.Compose([
        transforms.Resize((320, 320)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 归一化
    ])
    input_image = tfs(val_image).unsqueeze(0)
    # 加载模型参数并移至GPU
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.cuda()

    # 进行推理
    with torch.no_grad():
        out = net(input_image.cuda())
        pred = out.argmax(dim=1).squeeze().cpu().numpy()
        pred = np.expand_dims(pred, axis=0)
        colormap = np.array(colormap).astype('uint8')
        val_pre = colormap[pred]
    val_image = val_image.resize((320, 320))
    fig, ax = plt.subplots(1, 2, figsize=(9, 5))
    ax[0].imshow(val_image)
    ax[1].imshow(val_pre.squeeze())
    ax[0].axis('off')
    ax[1].axis('off')
    save_path = HISTORY_PATH + 'pic_{}_{}'.format(model_e, image_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    pre_img = Image.fromarray(val_pre.squeeze())
    pre_path = PRE_path + 'pic_{}_{}'.format(model_e, image_name)
    pre_img.save(pre_path)
    return pre_path
