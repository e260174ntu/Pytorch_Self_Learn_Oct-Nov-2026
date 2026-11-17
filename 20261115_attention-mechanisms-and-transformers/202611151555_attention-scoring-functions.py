import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import math
import torch
from torch import nn
from d2l import torch as d2l

def masked_softmax(X, valid_lens): # 这里X的三个维度是(batch,query,key)
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1) # 这里去的第二个维度的长度是key的长度
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None] # mask是一个条件判断的布尔值
        X[~mask] = value # 把不满足mask, 即mask返回False的位置全部用负值大数代替, 满足的不会被改变
        return X
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape # 存一遍一开始的形状
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) # 在计算之前把前两个维度展开了,比如(2,3,4)变成(6,4)
        return nn.functional.softmax(X.reshape(shape), dim=-1) # 根据原始形状再折叠回去
print(masked_softmax(torch.rand(2, 3, 4), torch.tensor([2, 3])))
print(masked_softmax(torch.rand(2, 2, 5), torch.tensor([[1, 3], [2, 4]])))

Q = torch.ones((100, 3, 4))
K = torch.ones((100, 4, 6))
print(torch.bmm(Q, K).shape) # 100个尺寸为(3,4)和(4,6)的矩阵进行乘法

# 点积注意力计算快（直接用高度优化的 GPU 矩阵乘法）但要求维度相同，且在维度d时值容易两极分化 必须缩放 现在 Transformer 的绝对主流
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Shape of queries: (batch_size, queries, dimension)  dimention means the dimentions of vector for a character
        # Shape of keys: (batch_size, key-value pairs, dimension)
        # Shape of values: (batch_size, key-value pairs, value dimension)
        # Shape of valid_lens: (batch_size,) or (batch_size, queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # Shape of score: (batch_size, queries, key-value pairs)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of weight is same as score: (batch_size, queries, key-value pairs)
        return torch.bmm(self.dropout(self.attention_weights), values)
        # Shape of return: (batch_size, queries, value dimension)
queries = torch.normal(0, 1, (2, 1, 2))
keys = torch.normal(0, 1, (2, 10, 2))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([3, 6])
attention = DotProductAttention(dropout=0.5)
attention.eval() # In evaluation mode, weights will not be cleared and the rest will not time 2
print(attention(queries, keys, values, valid_lens).shape)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.savefig(Path(__file__).stem+'_DotProduct.png', dpi=150, bbox_inches='tight')   

# 加性注意力在早期的RNN翻译模型（如Bahdanau注意力）中非常流行，但现在一般只有在小规模数据或对可解释性有要求的场景才用（因为矩阵加法和tanh的计算开销比纯点积大）
class AdditiveAttention(nn.Module):  
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # shape of queries: (batch_size, queries, num_hiddens) 
        # shape of keys: (batch_size, key-value pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # After dimension expansion, Sum them up with broadcasting
        # shape of queries: (batch_size, queries, 1, num_hiddens) 
        # shape of keys: (batch_size, 1, key-value pairs, num_hiddens)
        # Shape of features: (batch_size, queries, key-value pairs, num_hiddens)
        features = torch.tanh(features) 
        scores = self.w_v(features).squeeze(-1)  
        # Shape of scores: (batch_size, queries, key-value pairs)
        # There is only one output of self.w_v, so we use squeeze to remove the last one-dimensional entry from the shape.
        #计算得到的score没有被扔掉,只是本来[[hid_1, hid_2...]]用(num_hiddens,1)大小的矩阵计算成了[[score]], 我们扔掉中括号变成[score],数字数量没变
        self.attention_weights = masked_softmax(scores, valid_lens) 
        # Shape of weight is same as score: (batch_size, queries, key-value pairs)
        return torch.bmm(self.dropout(self.attention_weights), values) 
        # Shape of values: (batch_size, key-value pairs, value dimension)
        # Shape of return: (batch_size, queries, value dimension)
queries = torch.normal(0, 1, (2, 1, 20))
keys = torch.normal(0, 1, (2, 10, 2))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([6, 4])
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens).shape)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.savefig(Path(__file__).stem+'_Additive.png', dpi=150, bbox_inches='tight') 