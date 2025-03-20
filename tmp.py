import os
from PIL import Image

def convert_to_png(file_path):
    try:
        img = Image.open(file_path)
        if not file_path.lower().endswith('.png'):
            base_name, ext = os.path.splitext(file_path)
            new_file_path = f"{base_name}.png"
            img.save(new_file_path, "PNG")
            print(f"Converted {file_path} to {new_file_path}")
            os.remove(file_path)
            print(f"Removed original file: {file_path}")
        else:
            print(f"{file_path} is already a PNG file.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            convert_to_png(file_path)

# 指定文件夹路径
training_dir = './data/training'
validation_dir = './data/validation'

# 处理训练集文件夹
print("Processing training directory...")
process_directory(training_dir)

# 处理验证集文件夹
print("Processing validation directory...")
process_directory(validation_dir)

print("Conversion complete.")