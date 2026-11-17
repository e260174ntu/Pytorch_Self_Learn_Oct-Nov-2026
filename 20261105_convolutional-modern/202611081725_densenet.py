import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
    
# 一个样例
blk = DenseBlock(2, 3, 10) # 这一步初始化之后就已经决定了是:第一次conv_block(3, 10) 第二次conv_block(13, 10) 这个决定与下一行的X
X = torch.randn(4, 3, 8, 8) # PyTorch处理图像任务时标准的四维张量格式永远是(Batch_Size, Channels, Height, Width)
Y = blk(X) # 这个时候torch.Size([4, 23, 8, 8])
print(Y.shape) # 通道数 23 = 3 + 10*2，这个函数循环两次 第一次cat合并: 3+10=13层 给下一次conv_block 第二次cat合并: 13+10=23层 输出结果
# 由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。而过渡层可以用来控制模型复杂度。
blk = transition_block(23, 10)
print(blk(Y).shape)

#用densenet复刻resnet算法
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
num_channels, growth_rate = 64, 32 # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate # 上一个稠密块的输出通道数
    if i != len(num_convs_in_dense_blocks) - 1: # 在稠密块之间添加一个转换层，使通道数量减半
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight')