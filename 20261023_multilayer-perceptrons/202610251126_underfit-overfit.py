import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间作为高次多项式的系数
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 实际生成时使用的的系数
#真实公式是：y= 5 + 1.2*x − 3.4*x^2 + 5.6*x^3 + 噪声(正态分布的随机数)
features = np.random.normal(size=(n_train + n_test, 1)) # 用正态分布生成200行1列的二维矩阵，不用一维(200,)是为了维度对齐
np.random.shuffle(features) # 打乱顺序（shuffle）保证接下来的训练集和测试集随机混合
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) 
#计算(200,1)矩阵的(0,1,2,3,...,19)次方 由于次方数矩阵是(1,20) 得到的poly_features 的形状是 (200, 20)

for i in range(max_degree): #如果不除以阶乘缩小 在模拟过拟合的时候计算量直接爆掉
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)! 每一个次方大数除以阶乘缩小
labels = np.dot(poly_features, true_w) # 用(200,20)点乘(20,)得到的labels的维度是(200,)的一维数组
labels += np.random.normal(scale=0.1, size=labels.shape) #全部加上一个随机正态噪声

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]] 

print(features[:2], poly_features[:2, :], labels[:2]) #验证转换成功

def evaluate_loss(net, data_iter, loss):  #评估给定数据集上模型的损失 已存储在d2l
    metric = d2l.Accumulator(2)  # 两个维度分别是损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels,num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 不设置偏置bias，因为我们已经在多项式中实现了它
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',xlim=[1, num_epochs], ylim=[1e-3, 1e2],legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),d2l.evaluate_loss(net, test_iter, loss)))
    print('Real weight:',true_w[0:4])
    print('weight:', net[0].weight.data.numpy())
    
# 三阶多项式函数拟合(正常): 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], # 训练集输入：前100行的前4列特征（对应0次方到3次方）
      poly_features[n_train:, :4], # 测试集输入：后100行的前4列特征（对应0次方到3次方）
      labels[:n_train],            # 训练集标签：前100个样本的真实y值
      labels[n_train:])            # 测试集标签：后100个样本的真实y值
d2l.plt.savefig(Path(__file__).stem+'_good.png', dpi=150, bbox_inches='tight')

# 线性函数拟合(欠拟合): 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], 
      poly_features[n_train:, :2],
      labels[:n_train], 
      labels[n_train:])
d2l.plt.savefig(Path(__file__).stem+'_underfit.png', dpi=150, bbox_inches='tight')

# 高阶多项式函数拟合(过拟合): 从多项式特征中选取所有维度
train(poly_features[:n_train, :], 
      poly_features[n_train:, :],
      labels[:n_train], 
      labels[n_train:], 
      num_epochs=1500)
d2l.plt.savefig(Path(__file__).stem+'_overfit.png', dpi=150, bbox_inches='tight')

