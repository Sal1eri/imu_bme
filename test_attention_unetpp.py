import torch
import torch.nn.functional as F
from model.AttentionUNetPlusPlus import AttentionUNetPlusPlus

def test_model_output():
    # 创建模型
    model = AttentionUNetPlusPlus(in_channels=3, num_classes=2)
    
    # 设置不同的模式
    print("测试训练模式:")
    model.train()
    
    # 创建随机输入
    x = torch.randn(4, 3, 512, 512)
    
    # 前向传播
    output_train = model(x)
    
    # 检查输出类型
    print(f"输出类型: {type(output_train)}")
    
    if isinstance(output_train, list):
        print(f"深度监督输出数量: {len(output_train)}")
        print(f"第一个输出形状: {output_train[0].shape}")
        
        # 测试softmax能否正常工作
        try:
            for i, out in enumerate(output_train):
                softmax_out = F.log_softmax(out, dim=1)
                print(f"第{i+1}个输出的softmax形状: {softmax_out.shape}")
        except Exception as e:
            print(f"应用softmax时出错: {e}")
    else:
        print(f"输出形状: {output_train.shape}")
        
        # 测试softmax能否正常工作
        try:
            softmax_out = F.log_softmax(output_train, dim=1)
            print(f"Softmax后形状: {softmax_out.shape}")
        except Exception as e:
            print(f"应用softmax时出错: {e}")
    
    print("\n测试评估模式:")
    model.eval()
    
    # 前向传播
    output_eval = model(x)
    
    # 检查输出类型
    print(f"输出类型: {type(output_eval)}")
    
    if isinstance(output_eval, list):
        print(f"深度监督输出数量: {len(output_eval)}")
        print(f"第一个输出形状: {output_eval[0].shape}")
    else:
        print(f"输出形状: {output_eval.shape}")
        
        # 测试softmax能否正常工作
        try:
            softmax_out = F.log_softmax(output_eval, dim=1)
            print(f"Softmax后形状: {softmax_out.shape}")
        except Exception as e:
            print(f"应用softmax时出错: {e}")
    
    # 测试关闭深度监督
    print("\n测试关闭深度监督:")
    model.deep_supervision = False
    model.train()  # 即使在训练模式下也不使用深度监督
    
    # 前向传播
    output_no_ds = model(x)
    
    # 检查输出类型
    print(f"输出类型: {type(output_no_ds)}")
    
    if isinstance(output_no_ds, list):
        print(f"深度监督输出数量: {len(output_no_ds)}")
    else:
        print(f"输出形状: {output_no_ds.shape}")
        
        # 测试softmax能否正常工作
        try:
            softmax_out = F.log_softmax(output_no_ds, dim=1)
            print(f"Softmax后形状: {softmax_out.shape}")
        except Exception as e:
            print(f"应用softmax时出错: {e}")

if __name__ == "__main__":
    test_model_output() 