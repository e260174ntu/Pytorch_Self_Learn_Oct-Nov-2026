import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  # 构造一个PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter))) #打印第一个小批量样本检验读取过程正常
# 训练开始
from torch import nn  # nn是神经网络的缩写
net = nn.Sequential(nn.Linear(2, 1))
#Sequential 的价值在「多层」时体现：单层时它只是「多套了一层壳」，显得冗余；多层时它让你不用写 forward，一行 Sequential(...) 就定义完整网络。
#d2l 用 Sequential 是为了和后面章节的多层网络统一写法——你现在单层，后面会变成 2 层、3 层，代码结构不用改。
net[0].weight.data.normal_(0, 0.01) #初始化两个w
net[0].bias.data.fill_(0) #初始化b
#Linear(in,out)里面就只有weight和bias两种东西，两个数字表示输入层数和输出层数
#weight(out,in)表示每一个输入和输出之间的权重对应关系，而bias(out,)表示每一个输出需要叠加上得偏移量
loss = nn.MSELoss() #返回所有样本损失的平均值，已经除以 batch
trainer = torch.optim.SGD(net.parameters(), lr=0.03) #把 net 的所有参数（weight、bias）的引用存进优化器，并记下 lr=0.03。
#此刻 net 的 weight 还是初始化的那个随机值。真正的优化只在每次 trainer.step() 时发生，而且每一次step()都是完全对等的
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)      # 1.建计算图：l.grad_fn 挂着一整条链，l 持有它
        l.backward()             # 2.grad+=本轮梯度（初始为 0，因为上一轮末尾清过）
        trainer.step()           # 3.用 grad 更新
        trainer.zero_grad()      # 4.为下一轮清零
                                 # 以上顺序也可以更改为4-1-2-3或者1-4-2-3
    l = loss(net(features), labels) #一轮训练结束计算一个成果并打印
    print(f'epoch {epoch + 1}, loss {l:f}')