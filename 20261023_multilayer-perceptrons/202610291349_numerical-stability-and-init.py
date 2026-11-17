import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight');

M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))
print('乘以100个矩阵后\n', M)



class DeepMLP(nn.Module):  # 定义一个 20 层的深层网络（没有激活函数，纯线性，为了验证公式 4.8.5）
    def __init__(self, num_layers, layer_size):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            # 假设每层输入输出维度都是 layer_size (比如 256)
            layers.append(nn.Linear(layer_size, layer_size))
            # 如果你用 ReLU，其实应该用 Kaiming 初始化，这里为了对应书上的推导，先不加激活函数
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ==========================================
# 方案 A：PyTorch 默认初始化（不是 Xavier）
# ==========================================
print("=== 方案 A：默认初始化（可能会爆炸或消失） ===")
net_default = DeepMLP(num_layers=20, layer_size=256)
# 输入数据：均值 0，方差 1（即 γ² = 1）
x = torch.randn(128, 256) 
out = net_default(x)
print(f"输入数据的标准差: {x.std().item():.4f}")
print(f"第 20 层输出的标准差: {out.std().item():.4f}\n")


# ==========================================
# 方案 B：Xavier 正态分布初始化
# ==========================================
print("=== 方案 B：Xavier 正态分布初始化（信号保持平稳） ===")
def init_weights_xavier_normal(m):
    if isinstance(m, nn.Linear): # 或 type(m) == nn.Linear
        nn.init.xavier_normal_(m.weight) # 核心魔法：应用 Xavier 正态分布
        nn.init.zeros_(m.bias) # 偏置通常初始化为 0
net_xavier = DeepMLP(num_layers=20, layer_size=256)
net_xavier.apply(init_weights_xavier_normal)
out = net_xavier(x)
print(f"输入数据的标准差: {x.std().item():.4f}")
print(f"第 20 层输出的标准差: {out.std().item():.4f}\n")


# ==========================================
# 方案 C：Xavier 均匀分布初始化
# ==========================================
print("=== 方案 C：Xavier 均匀分布初始化（信号保持平稳） ===")
def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight) # 核心魔法：应用 Xavier 均匀分布
        nn.init.zeros_(m.bias) # 偏置通常初始化为 0
net_xavier_uni = DeepMLP(num_layers=20, layer_size=256)
net_xavier_uni.apply(init_weights_xavier_uniform)
out = net_xavier_uni(x)
print(f"输入数据的标准差: {x.std().item():.4f}")
print(f"第 20 层输出的标准差: {out.std().item():.4f}")