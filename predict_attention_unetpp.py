import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from model.AttentionUNetPlusPlus import AttentionUNetPlusPlus
import matplotlib.pyplot as plt

# 自定义转换函数，避免使用原始的utils/transform.py中的实现
class SimpleResize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        return image.resize(self.size, Image.BILINEAR)

class SimpleToTensor:
    def __call__(self, image):
        # 直接将PIL图像转换为tensor，不返回元组
        return torch.from_numpy(np.array(image).transpose((2, 0, 1)) / 255.0).float()

# 设置固定路径
MODEL_PATH = './model_result/best_attention_unetpp.pth'
INPUT_PATH = 'data/validation/images'  # 测试图像目录
OUTPUT_PATH = './prediction_results'  # 预测结果保存目录
INPUT_SIZE = (512, 512)  # 输入图像大小

def process_image(image_path):
    """处理单张图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 应用简单变换，避免使用原始transform
    image = SimpleResize(INPUT_SIZE)(image)
    image_tensor = SimpleToTensor()(image)
    
    return image_tensor, original_size

def predict_single_image(model, image_tensor, device):
    """对单张图像进行预测"""
    model.eval()
    with torch.no_grad():
        # 添加batch维度并移动到指定设备
        image = image_tensor.unsqueeze(0).to(device)
        
        # 模型预测
        output = model(image)
        
        # 如果输出是列表（深度监督），使用最后一个输出
        if isinstance(output, list):
            output = output[-1]
            
        # 获取预测结果
        pred = F.softmax(output, dim=1)
        pred = pred.argmax(dim=1).squeeze().cpu().numpy()
        return pred

def save_prediction(pred, original_size, output_path):
    """保存预测结果"""
    # 调整预测结果到原始图像大小
    pred = Image.fromarray(pred.astype(np.uint8))
    pred = pred.resize(original_size, Image.NEAREST)
    
    # 保存为二值图像
    pred.save(output_path)

def visualize_prediction(image_path, pred_path, output_path):
    """可视化预测结果"""
    # 读取原始图像
    image = Image.open(image_path).convert('RGB')
    
    # 读取预测结果
    pred = Image.open(pred_path)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原始图像
    ax1.imshow(image)
    ax1.set_title('原始图像')
    ax1.axis('off')
    
    # 显示预测结果
    ax2.imshow(pred, cmap='gray')
    ax2.set_title('预测结果')
    ax2.axis('off')
    
    # 保存可视化结果
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'visualizations'), exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = AttentionUNetPlusPlus(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
    model = model.to(device)
    
    # 获取输入文件列表
    input_files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"开始预测，共{len(input_files)}个文件...")
    
    for image_path in tqdm(input_files):
        try:
            # 获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # 处理图像
            image_tensor, original_size = process_image(image_path)
            
            # 预测
            pred = predict_single_image(model, image_tensor, device)
            
            # 保存预测结果
            pred_path = os.path.join(OUTPUT_PATH, 'predictions', f'{filename}_pred.png')
            save_prediction(pred, original_size, pred_path)
            
            # 保存可视化结果
            vis_path = os.path.join(OUTPUT_PATH, 'visualizations', f'{filename}_vis.png')
            visualize_prediction(image_path, pred_path, vis_path)
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            continue
    
    print(f"预测完成！结果保存在: {OUTPUT_PATH}")

if __name__ == '__main__':
    main() 