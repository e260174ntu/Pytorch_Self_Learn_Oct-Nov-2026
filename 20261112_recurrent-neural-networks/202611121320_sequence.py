import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

# 先生成一些数据标签
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.savefig(Path(__file__).stem+'_1.png', dpi=150, bbox_inches='tight')

# 准备开始训练
tau = 4 # 构建马尔可夫链 每一个预测只于前面四个有关
features = torch.zeros((T - tau, tau)) # 建一个(996,4)的空白矩阵
for i in range(tau):
    features[:, i] = x[i: T - tau + i] # (0~995,0)是x[0~995] (0~995,1)是x[1~996] (0~995,2)是x[2~997] (0~995,3)是x[3~998]
labels = x[tau:].reshape((-1, 1)) # labels把x[4~999],即要预测的答案,重构为形状为(996,1)的矩阵 
batch_size, n_train = 16, 600 # 只有前n_train个样本用于训练
loss = nn.MSELoss(reduction='none') # 平方损失。注意：MSELoss计算平方误差时不带系数1/2
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
def init_weights(m): # 初始化网络权重的函数
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def get_net(): # 一个简单的多层感知机
    net = nn.Sequential(nn.Linear(4, 10),  # 在这个训练过程中,每一个预测输出会有四个输入,4 -> 10 -> 1
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights) # 初始化这两个全连接层
    return net
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

# 以下将实现: 预测第5-1000个数据,每一个预测都是根据它的前四个数据进行
net = get_net()
train(net, train_iter, loss, 5, 0.01)
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.savefig(Path(__file__).stem+'_2.png', dpi=150, bbox_inches='tight')

# 以下将实现: 从第605开始, 根据预测的前四个来预测下一个, 而非根据实际的前四个来预测下一个
# 比如第605将根据预测的第601-604来预测
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))   
d2l.plt.savefig(Path(__file__).stem+'_3.png', dpi=150, bbox_inches='tight')     

# 以下将实现: 预测数据,每一个预测都是根据它之前的部分数据进行
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau): # 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
    features[:, i] = x[i: i + T - tau - max_steps + 1]
for i in range(tau, tau + max_steps): # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
    steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.savefig(Path(__file__).stem+'_4.png', dpi=150, bbox_inches='tight')   
