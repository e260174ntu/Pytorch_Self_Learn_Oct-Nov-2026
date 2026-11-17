import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
print(net(X))

class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self): 
        super().__init__() # 通过super调用MLP的父类Module的构造函数来初始化，这样在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层
    def forward(self, X): # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
        return self.out(F.relu(self.hidden(X))) # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
net = MLP()
print(net(X))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args): # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            self._modules[str(idx)] = module # 变量_modules中。_module的类型是OrderedDict
    def forward(self, X):
        for block in self._modules.values(): # OrderedDict保证了按照成员添加的顺序遍历它们
            X = block(X)
        return X
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.linear = nn.Linear(20, 20)
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1) # 使用创建的常量参数以及relu和mm函数
        X = self.linear(X) # 复用全连接层。这相当于两个全连接层共享参数
        # 控制流
        while X.abs().sum() > 1:
            X /= 2 #如果输入的数值很大，循环会执行很多次；如果数值小，可能一次都不执行。
        return X.sum()
net = FixedHiddenMLP()
print(net(X))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    def forward(self, X):
        return self.linear(self.net(X))
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))