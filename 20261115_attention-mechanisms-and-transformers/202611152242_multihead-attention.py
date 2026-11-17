import torch
from torch import nn
from torch.nn import functional as F
import math
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,heads):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.cache_k = None
        self.cache_v = None
    def forward(self, x, mask=None, cache=False):
        B, L, D = x.shape
        Q = self.Wq(x) # Q.shape: (B, L, d_model)
        K = self.Wk(x) # K.shape: (B, L, d_model)
        V = self.Wv(x) # V.shape: (B, L, d_model)
        Q = Q.view(B, L, self.heads, self.d_k).transpose(1, 2) # Q.shape: (B, heads, L, d_k)
        K = K.view(B, L, self.heads, self.d_k).transpose(1, 2) # K.shape: (B, heads, L, d_k)
        V = V.view(B, L, self.heads, self.d_k).transpose(1, 2) # V.shape: (B, heads, L, d_k)
        if cache:
            if self.cache_k is None:
                self.cache_v = V # cache_v.shape: (B, heads, L, d_k)
                self.cache_k = K # cache_v.shape: (B, heads, L, d_k)
            else:
                self.cache_k = torch.cat([self.cache_k, K], dim=-2)
                self.cache_v = torch.cat([self.cache_v, V], dim=-2)
            K = self.cache_k
            V = self.cache_v
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e6)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.Wo(out)
        return out