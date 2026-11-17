import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l
"""训练集只有 20 个样本(n_train = 20)。
测试集有 100 个样本(n_test = 100)。
特征维度高达 200 维(num_inputs = 200)。
在数学上, 这意味要解一个方程组:Xw=y。
你的 X矩阵是 (20, 200) 的。你要求解的 w向量是 (200, 1) 的。
这相当于:你只有 20 个已知条件, 却要解出 200 个未知数。
这在数学上是无穷多组解的(这叫欠定方程组)。"""
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    #如果net,loss=d2l.linreg(X,w,b),d2l.squared_loss 这里d2l.linreg会立刻执行,返回一个装满了具体数字的张量。那么net就是一个Tensor而不是一个函数。
    '''
    net = lambda X: d2l.linreg(X, w, b)等价于新建一个函数:
    def net(X):
        return d2l.linreg(X, w, b)
    '''
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项, 
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是:', torch.norm(w).item())
    d2l.plt.savefig(Path(__file__).stem+'_lambd='+str(lambd)+'.png', dpi=150, bbox_inches='tight');
    
train(lambd=0)
'''因此, 模型在训练时, 为了让这 20 个点的 Loss 降到极低, 它可以把这 200 个权重里的某些值变得极其离谱、极其巨大, 只要它们正负抵消, 能凑出这 20 个点的结果就行。
这就导致训练集 Loss 飞速下降(图 1 的蓝色实线掉到 10^-4以下)
但是换到 100 个测试样本时, 由于那 200 个权重全是在瞎凑, 完全没学到真实规律, 测试集 Loss 直接原地爆炸(图 1 的紫色虚线坚挺在 10^2左右)。
这就是你看到的第一张图。 特征极多 + 样本极少 = 必然过拟合。'''
train(lambd=3)
'''l2_penalty(w) 计算的是 200 个权重的平方和。
在第一张图(lambd=0)里, 因为没有惩罚, 模型为了凑那 20 个点, 把某些权重 w 撑得极其巨大。
现在你把 lambd=3 传进去, 惩罚项放大了 3 倍。如果模型还想像之前那样, 把某个权重放大到 1000 去凑答案, 
那么1000^2×3=3,000,000。这个巨大的值会直接让总损失 l 爆炸。
梯度下降算法为了保住总损失不爆炸, 只能被迫妥协:
强制把 w 压小:它不敢再把某些权重搞得忽大忽小, 只能让这 200 个权重尽量都趋近于均匀且较小的值(实际上真实的 true_w 就是全部为 0.01 的均匀值)。
放弃拟合噪声:它不再去强行穿过那 20 个训练点, 而是找一条相对平滑的面。
结果就是第二张图:加了惩罚后, 模型被“逼”着学出了接近真实的 0.01 权重
所以它的测试集 Loss(紫色虚线)随着训练一路下降, 最终和训练集 Loss 贴在一起。'''
train(lambd=8)
