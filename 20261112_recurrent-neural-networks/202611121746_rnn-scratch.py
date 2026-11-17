import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
# 修复 d2l 库中 load_corpus_time_machine 的一维展平 Bug
def fixed_load_corpus_time_machine(max_tokens=-1):
    lines = d2l.read_time_machine()
    tokens = d2l.tokenize(lines, 'char')
    # 核心修复：手动把二维列表展平成一维
    tokens_flat = [token for line in tokens for token in line]
    vocab = d2l.Vocab(tokens_flat)
    corpus = [vocab[token] for token in tokens_flat]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
# 用我们写好的函数，替换掉库里的有 Bug 的函数
d2l.load_corpus_time_machine = fixed_load_corpus_time_machine

def get_params(vocab_size, num_hiddens, device): # 初始化模型参数
    num_inputs = num_outputs = vocab_size
    # 隐藏层参数
    W_xh = torch.randn(size=(num_inputs, num_hiddens), device=device) * 0.01
    W_hh = torch.randn(size=(num_hiddens, num_hiddens), device=device) * 0.01
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = torch.randn(size=(num_hiddens, num_outputs), device=device) * 0.01
    b_q = torch.zeros(num_outputs, device=device)
    # 给所有的参数张量附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
def init_rnn_state(batch_size, num_hiddens, device): # 在初始化时返回隐状态
    return (torch.zeros((batch_size, num_hiddens), device=device), )
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs: #使用 tanh 函数作为激活函数
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,) # 最基础的RNN只有一个记忆球 隐藏状态H 如果未来手写 LSTM，那么记忆球会有两个（隐藏状态 H 和细胞状态 C）
class RNNModelScratch: #定义了所有需要的函数之后创建一个类来包装这些函数，并存储从零开始实现的循环神经网络模型的参数。
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def grad_clipping(net, theta):  #theta是人为设置的梯度大小阈值,是一个超参数
    """裁剪梯度:如果模长太大,找过预阈值,就在不改变方向的情况下缩短模长为阈值大小"""
    if isinstance(net, nn.Module): #isinstance(对象,类) 如果对象是目标类就返回True, 否则False
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params: # 循环给梯度的每个方向都等比例缩短,使得方向不改变,模长为阈值
            param.grad[:] *= theta / norm
            
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

def main():
    #这里使用独热编码的目的是消除不同字母之间的权重关系,如果按照1-26排列字母表,那么z的价值就会是a的26倍,这显然是不对的
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T, 28).shape)
    
    # 检查输出是否具有正确的形状
    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)
    
    # 训练之前只会输出乱码
    print("[Before training]",predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))
    
    #开始训练
    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu()) # 顺序抽样
    d2l.plt.savefig(Path(__file__).stem+'_sequence.png', dpi=150, bbox_inches='tight')
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True) # 随机抽样
    d2l.plt.savefig(Path(__file__).stem+'_random.png', dpi=150, bbox_inches='tight')
    
if __name__ == '__main__':
    main()