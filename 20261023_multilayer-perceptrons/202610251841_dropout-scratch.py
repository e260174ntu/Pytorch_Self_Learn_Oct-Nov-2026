import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1 #用assert来确保一个条件，条件不符程序直接报错
    if dropout == 1:
        return torch.zeros_like(X) # 在本情况中，所有元素都被丢弃
    if dropout == 0:
        return X # 在本情况中，所有元素都被保留
    mask = (torch.rand(X.shape) > dropout).float() 
    return mask * X / (1.0 - dropout) #为了确保期望一样，剩下的每一个都会被放大

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        #self.training = is_training AI说吧这一行去掉,咱也不敢问
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    def forward(self, X):  #PyTorch在底层硬编码了forward这个名字,Net(X)默认就是执行这个类里面的名叫forward(X)的函数
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training == True: # 只有在训练模型时才使用dropout
            H1 = dropout_layer(H1, dropout1)  # 在第一个全连接层之后添加一个dropout层
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)  # 在第二个全连接层之后添加一个dropout层
        out = self.lin3(H2)
        return out
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)