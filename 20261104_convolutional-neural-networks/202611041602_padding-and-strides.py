import torch
from torch import nn

# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
X = torch.rand(size=(8, 8))
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
print(comp_conv2d(conv2d, X).shape)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape) #补全变成10x10,123,345,567,789,4个
print(comp_conv2d(conv2d, torch.rand(size=(9, 9))).shape) #补全变成11x11,123,345,567,789,91011,5个