import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from u3plus.Qnet import ResNetUNet
from torchvision import transforms
import os
import skimage.morphology
import skimage.filters
import skimage.segmentation
import skimage.measure
from scipy import ndimage
import cv2
import pandas as pd
from math import pi
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_metrics(pred_mask, gt_mask):
    """
    计算准确率和IoU
    """
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    
    # 计算准确率
    accuracy = np.mean(pred_mask == gt_mask)
    
    # 计算IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    
    return accuracy, iou

def preprocess_image(image_path, input_size=(320, 320)):
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    
    # 调整图片大小
    image = image.resize(input_size, Image.BILINEAR)
    
    # 转换为tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    
    image_tensor = transform(image)
    return image_tensor, image

def calculate_organoid_stats(labeled_mask):
    """
    计算类器官的统计信息
    """
    # 确保掩码是整数类型
    labeled_mask = labeled_mask.astype(np.int32)
    props = skimage.measure.regionprops(labeled_mask)
    stats = []
    
    for prop in props:
        # 计算等效直径 (基于面积的圆形直径)
        equivalent_diameter = 2 * np.sqrt(prop.area / pi)
        
        stats.append({
            '编号': prop.label,
            '面积': prop.area,
            '直径': equivalent_diameter,
            '偏心率': prop.eccentricity,
            '质心_y': prop.centroid[0],
            '质心_x': prop.centroid[1]
        })
    
    return pd.DataFrame(stats)

def post_process_mask(mask, min_area=100, fill_holes=True, smooth_edges=True):
    """
    对预测的掩码进行增强后处理
    """
    # 二值化
    binary_mask = mask > 0
    
    # 1. 初步形态学操作：开运算去除小噪点
    binary_mask = skimage.morphology.binary_opening(binary_mask)
    
    # 2. 标记连通区域
    labeled_mask = skimage.measure.label(binary_mask)
    props = skimage.measure.regionprops(labeled_mask)
    
    # 创建新的标记掩码
    final_labeled_mask = np.zeros_like(labeled_mask)
    current_label = 1
    
    # 3. 对每个区域单独处理
    for prop in props:
        if prop.area < min_area:
            continue
            
        # 获取当前区域的掩码
        region_mask = labeled_mask == prop.label
        
        # 3.1 使用距离变换来找到区域内的空洞
        distance = ndimage.distance_transform_edt(region_mask)
        
        # 3.2 对当前区域进行空洞填充
        filled_region = skimage.morphology.remove_small_holes(
            region_mask, 
            area_threshold=prop.area * 0.2,  # 相对于区域面积的空洞阈值
            connectivity=2
        )
        
        # 3.3 平滑边缘
        if smooth_edges:
            # 使用更大的结构元素进行闭运算
            selem = skimage.morphology.disk(3)
            filled_region = skimage.morphology.binary_closing(filled_region, selem)
            
            # 添加额外的平滑处理
            filled_region = skimage.morphology.binary_dilation(filled_region, skimage.morphology.disk(1))
            filled_region = skimage.morphology.binary_erosion(filled_region, skimage.morphology.disk(1))
        
        # 3.4 更新标记图
        final_labeled_mask[filled_region] = current_label
        current_label += 1
    
    # 4. 最终的形态学优化
    if smooth_edges:
        # 对每个标记区域单独进行平滑处理
        smoothed_mask = np.zeros_like(final_labeled_mask)
        for label in range(1, final_labeled_mask.max() + 1):
            binary_region = final_labeled_mask == label
            smoothed_region = skimage.filters.gaussian(binary_region.astype(float), sigma=0.5) > 0.5
            smoothed_mask[smoothed_region] = label
        final_labeled_mask = smoothed_mask
    
    return final_labeled_mask.astype(np.int32)

def predict_single_image(model_path, image_path, output_path=None, stats_output_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ResNetUNet(n_class=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 预处理图片
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 加载真值图片
    gt_path = image_path.replace('images', 'segmentations')
    gt_image = Image.open(gt_path).convert('L')
    gt_image = gt_image.resize((320, 320), Image.NEAREST)
    gt_mask = np.array(gt_image) > 0
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        pred_prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
        confidence = pred_prob[0, 1].cpu().numpy()
    
    # 后处理掩码
    labeled_mask = post_process_mask(pred, min_area=50)
    
    # 计算评估指标
    accuracy, iou = calculate_metrics(labeled_mask, gt_mask)
    
    # 计算统计信息
    stats_df = calculate_organoid_stats(labeled_mask)
    
    if stats_output_path:
        stats_df.to_csv(stats_output_path, index=False, encoding='utf-8-sig')
        print(f'统计信息已保存到: {stats_output_path}')
    
    # 将原图转换为numpy数组
    original_image = np.array(original_image)
    
    # 创建可视化结果
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 显示原始图片
    axes[0].imshow(original_image)
    axes[0].set_title('原始图片')
    axes[0].axis('off')
    
    # 显示真值图片
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('真值标签')
    axes[1].axis('off')
    
    # 显示带编号的标记图
    im = axes[2].imshow(labeled_mask, cmap='nipy_spectral')
    axes[2].set_title(f'预测结果\n准确率: {accuracy:.3f}\nIoU: {iou:.3f}')
    for _, row in stats_df.iterrows():
        axes[2].text(row['质心_x'], row['质心_y'], str(int(row['编号'])), 
                    color='white', ha='center', va='center')
    axes[2].axis('off')
    
    # 显示叠加结果
    overlay = original_image.copy()
    for label in range(1, labeled_mask.max() + 1):
        mask = labeled_mask == label
        overlay[mask] = overlay[mask] * 0.7 + np.array([0, 255, 0]) * 0.3
        if label in stats_df['编号'].values:
            row = stats_df[stats_df['编号'] == label].iloc[0]
            cv2.putText(overlay, str(int(label)), 
                       (int(row['质心_x']), int(row['质心_y'])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    axes[3].imshow(overlay)
    axes[3].set_title('叠加结果')
    axes[3].axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'结果图像已保存到: {output_path}')
    else:
        plt.show()
    
    # 打印汇总统计信息
    print("\n类器官统计信息:")
    print(f"总数量: {len(stats_df)}")
    print(f"平均面积: {stats_df['面积'].mean():.2f} 像素")
    print(f"平均直径: {stats_df['直径'].mean():.2f} 像素")
    print(f"平均偏心率: {stats_df['偏心率'].mean():.2f}")
    print(f"\n评估指标:")
    print(f"准确率: {accuracy:.3f}")
    print(f"IoU: {iou:.3f}")

if __name__ == "__main__":
    # 设置路径
    model_path = './model_result/best_model_Qnet.mdl'
    image_path = './data/validation/images/42.png'
    output_path = './pic_results/single_prediction.png'
    stats_output_path = './pic_results/organoid_stats.csv'
    
    # 运行预测
    predict_single_image(model_path, image_path, output_path, stats_output_path) 