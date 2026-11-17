import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs): #num_layers=2是GRU层数
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size) # 嵌入层 
        # 如果词表有10000个词, One-hot会把每个词变成一个长度为 10000 的向量，
        # 里面全是 0，只有一个 1。极其浪费内存，而且没有包含任何语义（所有词的距离都是相等的）
        # Embedding：将离散的词表索引映射为 embed_size 维的连续稠密向量，赋予词义特征
        # 每个词是一个长度为 8（或 256）的稠密向量。最关键的是，这个矩阵里的数字是可以被“训练”的！ 
        # 随着模型的学习，意思相近的词（比如“猫”和“狗”），它们对应的向量在空间里会靠得很近。
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout) # 核心 RNN 层（GRU）接收词向量，提取序列特征，num_layers 代表堆叠的层数
    def forward(self, X, *args):
        X = self.embedding(X) # 输出'X'的形状：(batch_size,num_steps,embed_size)
        # PyTorch 的 RNN/GRU 要求第一维必须是时间步(Time Step)
        # 所以要维度交换,将(batch_size,num_steps,embed_size)转为(num_steps,batch_size,embed_size)
        X = X.permute(1, 0, 2) #原本(0,1,2)调换顺序成(1,0,2),其实就是swap前两项
        
        # 把调整顺序为第一维为时间步的 X 输入GRU，若未提及初始状态 state，则 GRU 内部会自动初始化为全 0 张量
        output, state = self.rnn(X)
        # 此时output的形状:(num_steps,batch_size,num_hiddens) state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层：和编码器一样，把目标语言的词索引变成稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_size) 
        # 核心 RNN 层：注意这里输入维度是 embed_size + num_hiddens！
        # 因为我们要把“当前词向量”和“编码器传过来的上下文向量”在特征维度拼接起来喂给 GRU
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        # 输出层：把 GRU 的隐藏状态映射回词表大小，用来预测下一个词的概率分布
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, *args):
        # 编码器前向传播返回的是 (output, state) 元组
        # 这里取索引 [1]，就是把编码器最后一步的终极记忆 state 拿过来，作为解码器的初始状态（接力棒交接）
        return enc_outputs[1]
    def forward(self, X, state):
        # 查字典并交换维度：输入 (batch_size, num_steps) -> 查表 (batch_size, num_steps, embed_size) -> 换轴 (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2) 
        # 广播 context（上下文向量）：state[-1] 取的是编码器最后一层的隐藏状态，形状 (batch_size, num_hiddens)
        # 为什么要 repeat？因为我们要让这个上下文向量，跟当前时间步的 X 拼在一起（每个时间步都需要它）
        context = state[-1].repeat(X.shape[0], 1, 1) # repeat 之后变成形状：(num_steps, batch_size, num_hiddens)
        X_and_context = torch.cat((X, context), 2) # 在 dim=2 拼接：把“当前词向量”和“上下文向量”拼起来
        # X_and_context 形状变为：(num_steps, batch_size, embed_size + num_hiddens)
        
        # 把调整顺序为第一维为时间步的 X_and_context 输入 GRU 并获得输出，同时带上历史记忆 state
        output, state = self.rnn(X_and_context, state) # 此时output 形状：(num_steps, batch_size, num_hiddens)
        output = self.dense(output).permute(1, 0, 2) # 再次交换维度变回以batch_size为第一维, 方便后续直接算交叉熵损失
        return output, state

batch_size, num_steps = 4, 7
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2) # 实例化编码器，词表大小 10，词向量维度 8，隐藏单元 16，GRU 2 层
encoder.eval() # 切换到评估模式（关闭 dropout，虽然这里没用到，但是好习惯）
X = torch.zeros((batch_size, num_steps), dtype=torch.long) # 构造输入数据：4 句话，每句 7 个词，全部用 0（未知词）填充
output, state = encoder(X) # 前向传播，获取编码器的输出特征以及最终的上下文状态
print(output.shape, state.shape)
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2) # 实例化解码器，词表大小 10，词向量维度 8，隐藏单元 16，GRU 2 层
decoder.eval()
state = decoder.init_state(encoder(X)) # 把编码器喂给解码器的 init_state，拿到初始记忆
# 注意：这里的 X 虽然和编码器的 X 长得一样（都是 0 矩阵），但实际中它是目标语言的输入（比如法语的 <bos> 和前几个词）
output, state = decoder(X, state)
print(output.shape, state.shape)

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
X = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [9, 8, 7, 6, 5, 4, 3]]) # 例如，如果两个序列的有效长度（不包括填充词元）分别为 2 和 3
print(sequence_mask(X, torch.tensor([2, 3]))) #则第一个序列的第一项和第二个序列的前两项之后的剩余项将被清除为零。
X = torch.ones(2, 3, 4)
print(sequence_mask(X, torch.tensor([1, 2]), value=-1)) # 也可以使用此函数, 用指定的非零值来替换(屏蔽)最后几个轴上的所有项

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
loss = MaskedSoftmaxCELoss()
print(loss(torch.ones(4, 4, 10), torch.ones((4, 4), dtype=torch.long),torch.tensor([4, 2, 1, 0])))

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()	# 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
def bleu(pred_seq, label_seq, k):  #用于评估预测的结果
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

#训练开始
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight')

#预测开始
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')