import os
import shutil
import numpy as np
from PIL import Image
import random

def convert_npy_to_png(npy_path, png_path):
    """将npy标签转换为png格式，值为0和255"""
    label = np.load(npy_path)
    # 确保标签值为0和255
    label = (label > 0.5).astype(np.uint8) * 255
    # 转换为PIL图像
    img = Image.fromarray(label)
    # 保存为PNG
    img.save(png_path)

def process_brain_data():
    # 源目录
    brain_img_dir = 'data/brain/imgs'
    brain_label_dir = 'data/brain/labels'
    
    # 目标目录
    train_img_dir = 'data/training/images'
    train_label_dir = 'data/training/segmentations'
    val_img_dir = 'data/validation/images'
    val_label_dir = 'data/validation/segmentations'
    
    # 确保目标目录存在
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 获取所有jpg图像文件
    jpg_files = [f for f in os.listdir(brain_img_dir) if f.endswith('.jpg')]
    
    # 随机打乱文件列表
    random.shuffle(jpg_files)
    
    # 计算训练集和验证集的分割点（80%训练，20%验证）
    split_point = int(len(jpg_files) * 0.8)
    
    # 处理训练集
    for jpg_file in jpg_files[:split_point]:
        # 获取对应的标签文件名（将.jpg替换为.npy）
        label_file = jpg_file.replace('.jpg', '.npy')
        
        # 源文件路径
        src_img = os.path.join(brain_img_dir, jpg_file)
        src_label = os.path.join(brain_label_dir, label_file)
        
        # 目标文件路径
        dst_img = os.path.join(train_img_dir, jpg_file)
        dst_label = os.path.join(train_label_dir, jpg_file.replace('.jpg', '.png'))
        
        # 复制图像文件
        shutil.copy2(src_img, dst_img)
        
        # 转换并保存标签
        if os.path.exists(src_label):
            convert_npy_to_png(src_label, dst_label)
    
    # 处理验证集
    for jpg_file in jpg_files[split_point:]:
        # 获取对应的标签文件名
        label_file = jpg_file.replace('.jpg', '.npy')
        
        # 源文件路径
        src_img = os.path.join(brain_img_dir, jpg_file)
        src_label = os.path.join(brain_label_dir, label_file)
        
        # 目标文件路径
        dst_img = os.path.join(val_img_dir, jpg_file)
        dst_label = os.path.join(val_label_dir, jpg_file.replace('.jpg', '.png'))
        
        # 复制图像文件
        shutil.copy2(src_img, dst_img)
        
        # 转换并保存标签
        if os.path.exists(src_label):
            convert_npy_to_png(src_label, dst_label)

if __name__ == '__main__':
    process_brain_data() 