import torch

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X) # 在本情况中，所有元素都被丢弃
    if dropout == 0:
        return X # 在本情况中，所有元素都被保留
    mask = (torch.rand(X.shape) > dropout).float() 
    return mask * X / (1.0 - dropout)
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5)) #并非严格的50%保留,而是大概率有8个保留,也有可能有7/9个,这是二项分布问题
print(dropout_layer(X, 1.))
