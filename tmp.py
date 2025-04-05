import torch
import torch.nn as nn

# 模拟模型输出
logits = torch.randn(3, 5)  # 假设批量大小为 3，类别数为 5
target = torch.tensor([1, 0, 3])  # 真实标签

# 先进行 softmax 操作
softmax_output = torch.softmax(logits, dim=1)
# 对 softmax 输出取对数
log_softmax_output = torch.log(softmax_output)
# 使用 NLLLoss 计算损失
nll_loss = nn.NLLLoss()
loss_softmax = nll_loss(log_softmax_output, target)

print("使用 softmax 计算的损失:", loss_softmax.item())

# 直接使用 logsoftmax
logsoftmax_output = torch.log_softmax(logits, dim=1)
# 使用 NLLLoss 计算损失
loss_logsoftmax = nll_loss(logsoftmax_output, target)

print("使用 logsoftmax 计算的损失:", loss_logsoftmax.item())

# 直接使用 CrossEntropyLoss
ce_loss = nn.CrossEntropyLoss()
loss_ce = ce_loss(logits, target)

print("直接使用 CrossEntropyLoss 计算的损失:", loss_ce.item())