import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())
class MyLinear(nn.Module):
    def __init__(self, in_units, out_units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, out_units))
        self.bias = nn.Parameter(torch.randn(out_units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5, 3)
print(linear.weight)
print(linear(torch.rand(2, 5)))
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(8, 64)))