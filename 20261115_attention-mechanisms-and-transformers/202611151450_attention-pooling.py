import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

d2l.use_svg_display()
# Define some kernels
def gaussian(x):
    return torch.exp(-x**2 / 2)
def boxcar(x):
    return torch.abs(x) < 1.0
def constant(x):
    return 1.0 + 0 * x
def epanechikov(x):
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = torch.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    ax.plot(x.detach().numpy(), kernel(x).detach().numpy()) # 先 .detach() 剥离梯度计算图，再.numpy()转成普通数组
    ax.set_xlabel(name)
d2l.plt.show()
d2l.plt.savefig(Path(__file__).stem+'_1.png', dpi=150, bbox_inches='tight')   

def f(x):
    return 2 * torch.sin(x) + x
n = 40
x_train, _ = torch.sort(torch.rand(n) * 5)
y_train = f(x_train) + torch.randn(n)
x_val = torch.arange(0, 5, 0.1)
y_val = f(x_val)

def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1)) # 由于广播机制, 生成一个(x_train,x-val)的二维矩阵
    k = kernel(dists).type(torch.float32) # Each column/row corresponds to each query/key 把距离扔进核函数，算出注意力分数
    attention_w = k / k.sum(0) # Normalization over keys for each query 算出累一个的概率占比(x_train, x_val)
    y_hat = y_train@attention_w # 一维key标签张量(x_train,)和二维权重张量(x_train, x_val)相乘 得到每个Query的标签预测值,这个结果是一个一维张量(x_val,)
    # PyTorch会把一维张量(x_train,)转化为二维张量(1,x_train),与(x_train, x_val)乘出来的尺寸是(1,x_val),再转化为一维张量(x_val,)
    return y_hat, attention_w
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            pcm = ax.imshow(attention_w.detach().numpy(), cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
plot(x_train, y_train, x_val, y_val, kernels, names)
d2l.plt.savefig(Path(__file__).stem+'_2.png', dpi=150, bbox_inches='tight')   
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
d2l.plt.savefig(Path(__file__).stem+'_3.png', dpi=150, bbox_inches='tight')  

sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]
def gaussian_with_width(sigma):
    return (lambda x: torch.exp(-x**2 / (2*sigma**2)))
kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
d2l.plt.savefig(Path(__file__).stem+'_4.png', dpi=150, bbox_inches='tight') 
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
d2l.plt.savefig(Path(__file__).stem+'_5.png', dpi=150, bbox_inches='tight')  