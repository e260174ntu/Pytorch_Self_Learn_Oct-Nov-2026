import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,kernel_size=3, padding=1)
        # 这样的设计要求2个卷积层的输出与输入形状一样，从而使它们可以相加。
        # 如果想改变通道数，就需要引入一个额外的1×1卷积层来将输入变换成需要的形状后再做相加运算。
        if use_1x1conv: 
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        #这里init里面通过两轮卷积层+批量标准化+激活函数得到的是残差Y = fx-x
        if self.conv3:
            X = self.conv3(X)
        Y += X #加入本身的X得到正确的Y，因为模型训练残差的效果好于直接训练下一步结果
        return F.relu(Y)

#感受形状变化
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6) # PyTorch处理图像任务时标准的四维张量格式永远是(Batch_Size, Channels, Height, Width)
Y = blk(X)
print(Y.shape)
blk = Residual(3,6, use_1x1conv=True, strides=2) #输入输出通道数不一样，就需要打开1x1卷积层来做通道数量转换器
print(blk(X).shape)
blk = Residual(3,100, use_1x1conv=True, strides=3)
print(blk(X).shape)

#用resnet复刻vgg算法
def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) #
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2)) # *型号用来去除list的[]标识
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight')

"""整体架构
net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    Residual(64, 64),
                    Residual(64, 64),
                    Residual(64, 128, use_1x1conv=True, strides=2),
                    Residual(128, 128),
                    Residual(128, 256, use_1x1conv=True, strides=2),
                    Residual(256, 256),
                    Residual(256, 512, use_1x1conv=True, strides=2),
                    Residual(512, 512),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
"""