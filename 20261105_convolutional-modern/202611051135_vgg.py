import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
"""nn.Sequential 需要接收的是一个个独立的层，比如 nn.Sequential(Conv2d, ReLU, MaxPool2d)。
直接写 nn.Sequential(layers)传进去的是一整个“列表”nn.Sequential([Conv2d, ReLU, MaxPool2d])，网络会不知所措。
加上 * 写成 nn.Sequential(*layers)，就等于把列表的括号[]去掉,变成了 nn.Sequential(Conv2d, ReLU, MaxPool2d) """
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 循环生成卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        # 返回卷积层结构和扁平化操作指令
        *conv_blks, nn.Flatten(),
        # 返回全连接层结构
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
#每一个小括号里面是 (num_convs, out_channels)
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) # 8卷积层 + 3全连接层 = 11层
#conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)) # 13卷积层 + 3全连接层 = 16层
net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for i, blk in enumerate(net):
    X = blk(X)
    print(f'Block {i}: {blk.__class__.__name__}, output shape: {X.shape}')
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight');