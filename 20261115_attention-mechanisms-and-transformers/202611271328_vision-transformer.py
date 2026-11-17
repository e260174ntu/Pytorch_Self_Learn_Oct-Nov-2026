import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512): # channels: 512
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,stride=patch_size)
    def forward(self, X):
        return self.conv(X).flatten(2).transpose(1, 2)
        # self.conv(X) shape: (batch size, channels, img_size//patch_size, img_size//patch_size)
        # flatten(2) : from dim=2 to the end: flatten into one degree: (batch size, channels, patches)
        # transpose and output shape: (batch size, patches, channels)
        # the 96x96x3 patch embedding outputs 36 patches, each patch has 512 dimentions
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = torch.zeros(batch_size, 3, img_size, img_size)
print(patch_emb(X).shape) # (batch_size, (img_size//patch_size)**2, num_hiddens)

class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
X = torch.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
print(encoder_blk(X).shape) # X.shape

class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks): 
            # 此时self.blks里面装着3个独立的Transformer Block,在装填过程中，没有任何数据 X 参与计算，只是在内存里搭好网络结构 初始化权重参数
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))
    def forward(self, X):
        # 1. 图像切块与特征映射
        # 假设输入 X 的形状为: (batch_size, 3, img_size, img_size)
        # 经过卷积投影和展平后，形状变成: (batch_size, num_patches, num_hiddens)
        # 例如：(B, 36, 512)
        X = self.patch_embedding(X)
        
        # 2. 拼接 <cls> (Class) Token
        # self.cls_token 的形状是 (1, 1, num_hiddens)
        # 通过 .expand() 沿着 batch 维度复制，变成 (batch_size, 1, num_hiddens), 里面全是0
        # 使用 torch.cat 沿维度 1（序列长度）拼接到最前面
        # 拼接后的形状变成: (batch_size, 1(cls) + num_patches, num_hiddens)
        # 例如：(B, 37, 512)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        
        # 3. 加入随机的位置编码并做 Dropout (这个随机的位置编码可被学习)
        # self.pos_embedding 的形状是 (1, 1 + num_patches, num_hiddens)
        # 此时正好与 X 的形状完美匹配（自动广播到 batch_size），相加融合位置信息
        # 然后通过 Dropout 随机失活部分神经元，防止过拟合
        """Q: 为什么不直接self.pos_embedding = nn.Parameter(torch.randn(batch_size, num_steps, num_hiddens))
        A: 不管送进来多少张图片，它们所遵循的空间坐标系是完全一样的！
        如果你把 pos_embedding 定义成 (batch_size, num_steps, num_hiddens)，那意味着：
        图片 A 的左上角用一套位置参数。图片 B 的左上角用另一套完全不同的位置参数。
        这就彻底乱套了，不仅毫无意义，而且会导致模型根本学不到通用的空间规律（因为你每次换个 batch，图片对应的位置参数就变了）。
        所以，我们只定义一套全局共享的位置参数，形状就是 (1, num_steps, num_hiddens)。"""
        X = self.dropout(X + self.pos_embedding)
        
        # 4. 穿过多个 Transformer Encoder Blocks
        # 逐个送入堆叠的 ViT Block，进行多层的自注意力机制交互
        # 经过这些层后，每一个 token（包括 cls_token 和所有 patch）都融合了全局上下文信息
        # 形状保持不变，依然是: (batch_size, 1 + num_patches, num_hiddens)
        for blk in self.blks:
            X = blk(X)
            
        # 5. 提取 <cls> Token 特征并分类
        # X[:, 0] 取出序列的第一个位置（即 <cls> token），形状变成: (batch_size, num_hiddens)
        # 因为这个 token 通过自注意力吸收了整张图的所有信息，所以用它做图像分类最合适
        # self.head 通常包含 LayerNorm 和全连接层，最终输出形状: (batch_size, num_classes)
        return self.head(X[:, 0])
    
def main():
    img_size, patch_size = 96, 16
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                num_blks, emb_dropout, blk_dropout, lr)
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
    trainer.fit(model, data)
    d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight') 
    
if __name__ == '__main__':
    main()