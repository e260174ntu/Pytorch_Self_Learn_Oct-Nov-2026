import torch
from d2l import torch as d2l

# 首先,我们定义矩阵X  W_xh  H  W_hh 它们的形状分别为(3,1) (1,4) (3,4) (4,4)
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
# 将这两个乘法相加 我们得到一个形状为(3,4)的矩阵
A = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# 沿列（轴1）拼接矩阵`X`和`H`,沿行（轴0）拼接矩阵`W_xh`和`W_hh`。
# 这两个拼接分别产生形状(3, 5)和形状(5, 4)的矩阵。再将这两个拼接的矩阵相乘,我们得到与上面相同形状(3, 4)的输出矩阵
B = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))

# 这里只是想说明两次矩阵乘法可以合并成一次,优化计算速度,利用的是分块矩阵乘法
print(A,B,A==B,sep='\n')

#这里事项说明可以把加偏置的步骤也放进矩阵乘法
b_h = torch.zeros(1, 4) # 因此用0填充偏置
C = A + b_h
D = torch.matmul(torch.cat((X, H, torch.ones(3, 1)), 1), torch.cat((W_xh, W_hh, b_h), 0))
print(C, D, C==D, sep='\n')