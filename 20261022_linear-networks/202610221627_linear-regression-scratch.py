import matplotlib
matplotlib.use('Agg')
import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import random
from pathlib import Path
script_name = Path(__file__).stem
out_file = script_name + '.png'

def synthetic_data(w, b, num_examples):  # 生成y=Xw+b+噪声
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1)) #y.reshape((-1, 1)) = 把一维的 y 竖起来，变成列向量，-1 是让 PyTorch 自动算行数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # 生成1000个 y = 二阶标准正态分布随机矩阵X * 二阶矩阵w + 截距b + 标准正态分布偏移
#print('features:', features[0],'\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.savefig(out_file, dpi=150, bbox_inches='tight') #把这些根据答案（直线拟合参数）生成的一千个点放在一张图上

def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 1000，样本总数
    indices = list(range(num_examples))  # 把编号单独建立一个数组[0, 1, 2, ..., 999]
    random.shuffle(indices) # 打乱编号，使得每次取特定编号的这些样本相当于是随机读取的
    for i in range(0, num_examples, batch_size): #读取已经打乱的编号里的第0 10 20 ... 990号作为起点进行迭代
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])  #取打乱后下标一段列表存入张量，比如 indices[0:10] 是前 10 个随机下标 dtype=torch.int64
        #最后一组可能不足 batch_size。比如 n=1005、batch_size=10，最后一组 i=1000，i+10=1010 > 1005，min 把它截到 1005，所以取 indices[1000:1005]（5 个）——不会越界。
        yield features[batch_indices], labels[batch_indices] #yield 相当于是暂时不返回的return，这次结束会把全程所有yield的值全扔回去
        #从自变量库 (1000, 2) 里按行取出这 10 行 → 形状 (10, 2) 两列是features是两个特征值，即自变量
        #从因变量库 (1000, 1) 里按行取出这 10 行 → 形状 (10, 1) 这一是标签值，即输出值，因变量
def linreg(X, w, b):  
    return torch.matmul(X, w) + b # 线性回归模型
def squared_loss(y_hat, y):  
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  #计算均方损失
def sgd(params, lr, batch_size):   #小批量随机梯度下降并优化权重和偏移量
    with torch.no_grad():  #导入参数的时候顺带导入梯度附属值，但是函数内不可继续计算的梯度
        for param in params:  #先更新张量w（两项），再更新b张量，一共三次
            param -= lr * param.grad / batch_size # torch.Tensor是可变对象，可以直接改对象内部的数据，全局都能看到
            param.grad.zero_() #清空存储的导数梯度
batch_size = 10  #小批量一次10个
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) 
b = torch.zeros(1, requires_grad=True) #通过梯度拟合的w和b            
lr = 0.03  #学习度，一个人为设定的参数来作为梯度修正拟合结果的放大倍数
num_epochs = 10  #一共十轮迭代
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  # 以小批量数目为每一小组的点数，随机但完全遍历所有的点，计算这么多组的X和y的小批量损失
        l = squared_loss(linreg(X, w, b), y)  # X是(10,2) linreg()是(10,1)预测值张量 y(10,1)是实际值张量，计算这10个的偏差
        l.sum().backward() # 因为l形状是(batch_size,1)，而不是一个标量。l中的10个元素被加到一起，
        # 并以此用backward计算关于[w,b]的梯度，存入w.grad和b.grad
        sgd([w, b], lr, batch_size)  # 使用两个参数张量的梯度更新参数
    #案例中1000个样本张量，金币搞过一次随机打乱，每十个打包一小组，一共一百个小组，每一组都会计算差距并拟合一次w和b，所以在一轮训练中，拟合改良了一百次w和b
    with torch.no_grad(): #每一次训练结束，计算一下全体1000个的总偏差作为一次训练的成果
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, squared_loss {float(train_l.mean()):f}')
#每一次训练都会随机打乱，所以从头到尾打乱了10次，一共优化100*10=1000次
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
