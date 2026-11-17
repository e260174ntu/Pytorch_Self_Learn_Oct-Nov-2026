import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers, device= len(vocab), 256, 2 ,d2l.try_gpu()
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNLM(lstm_layer, len(vocab))
model = model.to(device)
num_epochs, lr = 500, 2.0
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device) 
d2l.plt.savefig(Path(__file__).stem+'.png', dpi=150, bbox_inches='tight')