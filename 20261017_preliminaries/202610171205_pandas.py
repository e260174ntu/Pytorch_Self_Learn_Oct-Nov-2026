import os
import numpy
import pandas
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pandas.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)
inputs = pandas.get_dummies(inputs, dummy_na=True)
print(inputs)
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X,'\n',Y)

a = torch.arange(12)
b = a.reshape((3,4)) # b 并没有复制 a ，而是创建了一个 a 的 view
print(a,'\n',b)
b[:] = 2
print(a,'\n',b)


# ---------------- 构造一个更大、更复杂的原始数据集 ----------------
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_large.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Age,Location,HasGarden,Price\n')
    f.write('NA,Pave,5,Downtown,NA,127500\n')
    f.write('2,NA,10,Uptown,Yes,106000\n')
    f.write('4,NA,3,Suburb,No,178100\n')
    f.write('NA,NA,7,Downtown,Yes,140000\n')
    f.write('3,Pave,NA,NA,No,155000\n')
    f.write('5,NA,2,Uptown,Yes,190000\n')
    f.write('NA,Pave,8,Suburb,NA,145000\n')
    f.write('1,NA,12,NA,No,90000\n')

data = pandas.read_csv(data_file)
print('原始数据:\n', data, '\n')

# ---------------- 1. 删除缺失值最多的列 ----------------
na_counts = data.isna().sum()
print('各列缺失值数量:\n', na_counts, '\n')

col_drop = na_counts.idxmax()          # 缺失值最多的列名
print(f'缺失值最多的列是 "{col_drop}"，共缺失 {na_counts[col_drop]} 个')
data = data.drop(columns=[col_drop])
print('\n删除该列后的数据:\n', data, '\n')

# ---------------- 2. 预处理 + 转换为张量 ----------------
inputs, outputs = data.iloc[:, :-1], data.iloc[:, -1]

# 数值列缺失 → 用均值填补
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print('数值列均值填补后:\n', inputs, '\n')

# 类别列 → one-hot 编码（NA 也作为一个类别）
inputs = pandas.get_dummies(inputs, dummy_na=True)
print('one-hot 编码后:\n', inputs, '\n')

# 转换为张量（统一为 float 类型，与 PyTorch 习惯一致）
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print('X =\n', X)
print('Y =\n', Y)