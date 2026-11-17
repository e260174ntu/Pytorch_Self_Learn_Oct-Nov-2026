import torch
import math

x=torch.arange(4.0,requires_grad=True) #[0,1,2,3]
print(x.grad)# 默认值是None
y = 2*torch.dot(x,x)
print(y) 
y.backward()
print(x.grad)
print(x.grad==4*x)


x.grad.zero_() # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
y = x.sum()
y.backward() #通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
print(x.grad)


# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
y.sum().backward() #直接求和所有元素，等价于y.backward(torch.ones(len(y)))
#也等价于y.backward(torch.ones(len(x)))
#y.backward(v) 表示dot(v,y),是一个标量
print(x.grad)

x.grad.zero_()
y = x * x
v = torch.tensor([1., 2., 3., 4.])
y.backward(v) # y标量=1*x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2
print(x.grad)    #梯度[2*x1,4*x2,6*x3,8*x4]=dot([2,4,6,8],[0,1,2,3])=[0,4,12,24]

x.grad.zero_()
y = x * x  #y张量=[0,1,4,9]
u = y.detach() # 张量u=y=[0,1,4,9] 但 u.requires_grad = False
ux = u * x #ux张量=[0,1,8,27] ux计算路径= 0*x1 + 1*x2 + 4*x3 + 9*x4
ux.sum().backward() #求和36返回到x.grad的是梯度偏导[0,1,4,9]
print(x.grad) #这与前面u张量[0,1,4,9]一模一样
x.grad.zero_()
yx = y * x #yx张量=[0,1,8,27] yx计算路径= x1^3 + x2^3 + x3^3 + x4^3
yx.sum().backward() #求和36返回到x.grad的是梯度偏导[3*x1^2, 3*x2^2, 3*x3^2, 3*x4^2]
print(x.grad) #[0,3,12,27]

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)
print(a.grad == d / a)
print("a =", a.item())
print("d =", d.item())
print("a.grad =", a.grad.item())
print("d/a    =", (d / a).item())
print("差值   =", (a.grad - d / a).item())
#torch.randn(size=())       tensor(-0.4321)            shape=torch.Size([])     ndim=0
#torch.randn(size=(1,))     tensor([-0.4321])          shape=torch.Size([1])    ndim=1
#torch.randn(size=(3,))     tensor([0.1, -0.5, 1.2])   shape=torch.Size([3])    ndim=1
#torch.randn(size=(2,3))    2×3 matrix                 shape=torch.Size([2, 3]) ndim=2
#torch.rand	均匀分布 U(0, 1)	[0, 1)	tensor([0.32, 0.87, 0.05])
#torch.randn	标准正态 N(0, 1)	(-∞, +∞)，均值 0，标准差 1	tensor([-0.43, 1.27, 0.08])
#torch.randint	整数均匀	你指定的 [low, high)	tensor([3, 0, 7])
#torch.randperm	0..n-1 的随机排列	—	tensor([2, 0, 3, 1])


import numpy as np
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from d2l import torch as d2l

def use_svg_display():  #@save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
    
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def f(x):
    return np.sin(x)

def numerical_grad(f, x, h=1e-5):
    """中心差分：f'(x) ≈ [f(x+h) - f(x-h)] / (2h)"""
    return (f(x + h) - f(x - h)) / (2 * h)

x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = f(x)
y_grad = numerical_grad(f, x)

plot(x, [y, y_grad], 'x', 'f(x)', 
     legend=['f(x) = sin(x)', "f'(x) ≈ numerical"],
     xlim=[-2*np.pi, 2*np.pi], ylim=[-1.5, 1.5])

d2l.plt.show()