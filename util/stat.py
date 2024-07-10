import torch
from torchinfo import summary
from model.unet_basic import Model
from thop import profile

model4stat = Model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model4stat.load_state_dict(torch.load("/home/yuhengfeng/Experiments/Wave-U-Net/train1/checkpoints/model_0450.pth"))
model4stat.to(device)
model4stat.eval()

total = sum([param.nelement() for param in model4stat.parameters()])
print('Number of parameter: % .4fM, % .f个' % (total / (2 ** 20), total))

summary(model4stat, (1, 1, 16384))

input_data = torch.randn(1, 1, 16384).to(device)
Flops, params = profile(model4stat, inputs=(input_data,))
print('Flops: % .4fG' % (Flops / (2 ** 30)))  # 计算量
print('params参数量: % .4fM' % (params / (2 ** 20)))
