import torch
import numpy as np
from PIL import Image
import os
import re
from u3plus.Qnet import ResNetUNet
from torchvision import transforms
import cv2
import skimage.morphology
import skimage.measure
from scipy import ndimage
import skimage.filters
import pandas as pd
import imageio

def preprocess_image(image_path, input_size=(320, 320)):
    """预处理图像以适应模型输入"""
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    
    # 保存原始尺寸，用于后续恢复
    original_size = image.size
    
    # 调整图片大小
    image = image.resize(input_size, Image.BILINEAR)
    
    # 转换为tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    
    image_tensor = transform(image)
    return image_tensor, image, original_size

def post_process_mask(mask, min_area=50):
    """对预测结果进行后处理"""
    # 二值化
    binary_mask = mask > 0
    
    # 初步形态学操作：开运算去除小噪点
    binary_mask = skimage.morphology.binary_opening(binary_mask)
    
    # 标记连通区域
    labeled_mask = skimage.measure.label(binary_mask)
    props = skimage.measure.regionprops(labeled_mask)
    
    # 创建新的标记掩码
    final_labeled_mask = np.zeros_like(labeled_mask)
    current_label = 1
    
    # 对每个区域单独处理
    for prop in props:
        if prop.area < min_area:
            continue
            
        # 获取当前区域的掩码
        region_mask = labeled_mask == prop.label
        
        # 使用距离变换来找到区域内的空洞
        distance = ndimage.distance_transform_edt(region_mask)
        
        # 对当前区域进行空洞填充
        filled_region = skimage.morphology.remove_small_holes(
            region_mask, 
            area_threshold=prop.area * 0.2,  # 相对于区域面积的空洞阈值
            connectivity=2
        )
        
        # 平滑边缘
        selem = skimage.morphology.disk(3)
        filled_region = skimage.morphology.binary_closing(filled_region, selem)
        
        # 添加额外的平滑处理
        filled_region = skimage.morphology.binary_dilation(filled_region, skimage.morphology.disk(1))
        filled_region = skimage.morphology.binary_erosion(filled_region, skimage.morphology.disk(1))
        
        # 更新标记图
        final_labeled_mask[filled_region] = current_label
        current_label += 1
    
    # 对每个标记区域单独进行平滑处理
    smoothed_mask = np.zeros_like(final_labeled_mask)
    for label in range(1, final_labeled_mask.max() + 1):
        binary_region = final_labeled_mask == label
        smoothed_region = skimage.filters.gaussian(binary_region.astype(float), sigma=0.5) > 0.5
        smoothed_mask[smoothed_region] = label
    
    return smoothed_mask.astype(np.int32)

def create_overlay_image(original_image, labeled_mask):
    """创建标记叠加在原图上的图像"""
    overlay = np.array(original_image).copy()
    
    # 为标记区域添加半透明绿色覆盖
    for label in range(1, labeled_mask.max() + 1):
        mask = labeled_mask == label
        # 使用绿色标记区域
        overlay[mask] = overlay[mask] * 0.7 + np.array([0, 255, 0]) * 0.3
    
    return overlay

def natural_sort_key(s):
    """实现自然排序的键函数"""
    # 提取文件名中的数字部分
    numbers = re.findall(r'\d+', s)
    if numbers:
        # 如果文件名中有数字，返回该数字的整数值
        return int(numbers[-1])
    return s

def predict_images(input_dir, output_dir, model_path):
    """对输入目录中的所有PNG图像进行预测，并保存结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    
    # 加载模型
    model = ResNetUNet(n_class=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载成功")
    
    # 寻找输入目录中的所有PNG文件并按自然顺序排序
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    
    # 使用自然排序
    image_paths.sort(key=natural_sort_key)
    print(f"找到 {len(image_paths)} 个PNG图像")
    print("图像处理顺序:")
    for path in image_paths[:5]:
        print(f"  - {os.path.basename(path)}")
    if len(image_paths) > 5:
        print("  ...")
    
    # 用于存储每张图片的预测结果数据
    results_data = []
    # 用于存储overlay图像用于生成GIF
    overlay_images = []
    
    # 处理每个图像
    for image_path in image_paths:
        # 提取文件名 (不带路径和扩展名)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"处理图像: {filename}")
        
        # 预处理图片
        image_tensor, original_image, original_size = preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
        
        # 后处理掩码
        labeled_mask = post_process_mask(pred)
        
        # 创建叠加图像
        overlay = create_overlay_image(original_image, labeled_mask)
        
        # 保存标记掩码
        mask_path = os.path.join(output_dir, "masks", f"{filename}_mask.png")
        cv2.imwrite(mask_path, labeled_mask * 50)  # 乘以50以增强可见度
        
        # 保存叠加图像
        overlay_path = os.path.join(output_dir, "overlays", f"{filename}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 添加到GIF列表
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_images.append(overlay_rgb)
        
        # 计算预测区域的总面积（像素数）
        total_area = np.sum(labeled_mask > 0)
        
        # 存储结果数据
        results_data.append({
            '图片名称': filename,
            '预测区域面积(像素)': total_area,
            '标记区域数量': labeled_mask.max()
        })
    
    # 生成GIF
    gif_path = os.path.join(output_dir, "prediction_sequence.gif")
    print("正在生成GIF动画...")
    imageio.mimsave(gif_path, overlay_images, duration=0.5)  # 每帧持续0.5秒
    
    # 保存结果到CSV
    csv_path = os.path.join(output_dir, "prediction_results.csv")
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"所有图像处理完成，结果保存在 {output_dir}")
    print(f"预测序列动画已保存为: {gif_path}")
    print(f"预测结果数据已保存为: {csv_path}")

if __name__ == "__main__":
    input_dir = "1L-40"
    output_dir = "1L-40_results"
    model_path = "./model_result/best_model_Qnet.mdl"
    
    print("开始使用Qnet模型预测1L-40中的图像")
    predict_images(input_dir, output_dir, model_path)
    print("预测完成！")
