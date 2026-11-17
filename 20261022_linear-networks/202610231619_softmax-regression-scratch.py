import torch
import torchvision
from torchvision import transforms
from IPython import display
from d2l import torch as d2l
from pathlib import Path
script_name = Path(__file__).stem
out_file = script_name + '.png'
batch_size = 128
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
lr = 0.1
num_epochs = 10

class Accumulator:  #在n个变量上累加
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx] 
def softmax(X):
    X_exp = torch.exp(X)                          # ① 指数化，保证正数
    partition = X_exp.sum(1, keepdim=True)        # ② 每一行求和，形状 (n,1)
    return X_exp / partition                      # ③ 广播除法，每行归一化
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) #把 4D 图像展平成 2D，再做线性变换 + softmax
def cross_entropy(y_hat, y):
    """从yhat里面拿出y的第0个和第1个所对应编号的元素 高级索引的高级索引
    y = torch.tensor([1, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(y_hat[[0, 1], y]) #我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率"""
    return - torch.log(y_hat[range(len(y_hat)), y]) 
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
def evaluate_accuracy(net, data_iter): 
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
def train_epoch_ch3(net, train_iter, loss, updater):  
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
def predict_ch3(net, test_iter, n=6):  #预测标签
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
def check():
    #---检验softmax正确性---
    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X) #随机一个2行5列的张量扔到softmax里面处理一下
    print(X_prob, X_prob.sum(1)) #每行总和应该为1
    #---从yhat里面拿出y的第0个和第1个所对应编号的元素（高级索引的高级索引）---
    y = torch.tensor([1, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(y_hat[[0, 1], y]) #我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率
    print(cross_entropy(y_hat, y))
    print(accuracy(y_hat, y) / len(y))
def main():
    #check()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) #返回两个 DataLoader,可以直接 for X, y in train_iter: 迭代
    print(evaluate_accuracy(net, test_iter))
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)
    d2l.plt.savefig(out_file, dpi=150, bbox_inches='tight')
if __name__ == '__main__':
    main()