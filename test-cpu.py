import torch
from models.experimental import attempt_load

# 强制使用CPU（即使有CUDA环境也禁用GPU）
device = torch.device('cpu')
model = attempt_load("yolov8n.pt", device=device)  # 加载模型到CPU

# 测试推理
img = torch.randn(1,3,640,640).to(device)  # 生成测试输入
with torch.no_grad():
    pred = model(img)[0]  # 成功执行表示CPU可用