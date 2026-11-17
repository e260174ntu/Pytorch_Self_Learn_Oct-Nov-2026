import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
d2l.plt.savefig(Path(__file__).stem+'1.png', dpi=150, bbox_inches='tight')

y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l.plt.savefig(Path(__file__).stem+'2.png', dpi=150, bbox_inches='tight')

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.savefig(Path(__file__).stem+'3.png', dpi=150, bbox_inches='tight')

x.grad.data.zero_() # 清除以前的梯度
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l.plt.savefig(Path(__file__).stem+'4.png', dpi=150, bbox_inches='tight')

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
d2l.plt.savefig(Path(__file__).stem+'5.png', dpi=150, bbox_inches='tight')

x.grad.data.zero_() #  清除以前的梯度
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.savefig(Path(__file__).stem+'6.png', dpi=150, bbox_inches='tight')