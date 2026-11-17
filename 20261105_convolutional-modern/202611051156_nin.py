import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(), #1x1的全连接层，输出数不变，所以输入出也就等于输出数就好
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
    
net = nn.Sequentia(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5), #原程序有这一行，但是卷积层一般参数较少，不太会使用dropout
    nin_block(384, 10, kernel_size=3, strides=1, padding=1), # 对于fashion-minst数据集，标签类别数是10
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()) # 将四维的输出转成二维的输出，原本形状是(批量大小,10,1,1), 扁平化之后的形状为(批量大小,10)
'''假设你现在手里有一张通道数为 10、大小为 5x5 的特征图 [1, 10, 5, 5]。
    当它经过 nn.AdaptiveAvgPool2d((1, 1)) 时：程序会把这 10 个通道里，每个 5x5 的矩阵，所有 25 个数字加起来除以 25 算出一个平均数。
    每个通道算出一个平均分 10 个通道就变成 10 个数字。最终输出变成了 [1, 10, 1, 1]。
    然后通过 nn.Flatten() 一展平，它就变成了一个长度为 10 的向量。这 10 个向量刚好对应 10 个类别，直接送去做 Softmax 分类即可。
    好处1 彻底消灭了参数量：传统的做法（如 VGG）是把特征图展平，然后用一个巨大的全连接层去分类（比如 16 * 5 * 5 -> 120）。全连接层参数量极大，极易过拟合。
          而 AdaptiveAvgPool2d 只是一个求平均的数学操作，不需要学习任何参数（参数量为 0）。
    好处2 解除了对输入图像尺寸的限制：如果你的模型接入了这个池化层，训练时输入 224x224 的图，推理时突然换成 256x256 的图，
          模型依然能正常工作，最后都会稳定输出 1x1 的向量。
    好处3 赋予了特征图“全局视野”：在展平之前，每个通道里的数字代表“某个局部区域有没有这个特征”。求平均后，这个数字就代表“整张图总体上有没有这个特征”。
          比如第 0 个通道代表“猫的耳朵”，全局平均后的值很高，就说明整张图里猫耳朵特征很强烈。'''
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
    
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight');