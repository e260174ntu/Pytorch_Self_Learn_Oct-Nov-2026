import torch
from torch import nn
from d2l import torch as d2l

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
num_epochs, lr, batch_size = 10, 0.5, 256
dropout1, dropout2 = 0.2, 0.5
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(dropout1), # 在第一个全连接层之后添加一个dropout层
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),# 在第二个全连接层之后添加一个dropout层
        nn.Linear(256, 10))
net.apply(init_weights) #apply可以将nn模块遍历,并依次对每一层调用函数
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
"""net.apply(init_weights)也可以写成如下形式:
def init_weights(net):
    for m in net.modules():  # 手动遍历所有子模块
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
init_weights(net)"""