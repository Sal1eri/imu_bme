import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("GPU (CUDA) is available!")
    # 打印可用的GPU设备数量
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    # 打印当前使用的GPU设备名称
    print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU (CUDA) is not available. Using CPU.")