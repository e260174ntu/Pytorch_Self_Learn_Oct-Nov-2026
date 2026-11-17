import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
from torch import nn
from d2l import torch as d2l

class AttentionDecoder(d2l.Decoder): 
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()
    @property
    def attention_weights(self):
        raise NotImplementedError
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)
    def init_state(self, enc_outputs, enc_valid_lens):
        outputs, hidden_state = enc_outputs
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
        # Shape of return: ((batch_size, num_steps, num_hiddens), hidden_state, enc_valid_lens)
    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        # Shape of enc_valid_lens: (batch_size)
        X = self.embedding(X).permute(1, 0, 2)
        # Shape of the output X: (num_steps, batch_size, embed_size)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1) 
            # Shape of query: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens) 
            # Shape of context: (batch_size, 1, num_hiddens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) 
            # Concatenate on the feature dimension
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state) 
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        # After fully connected layer transformation, shape of outputs: (num_steps, batch_size, vocab_size)
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]
        # shape of return: (batch_size, num_steps, vocab_size), [enc_outputs, hidden_state, enc_valid_lens]
    @property
    def attention_weights(self):
        return self._attention_weights
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
enc_valid_lens = torch.tensor([3, 5, 7, 2])
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens, num_layers)
X = torch.zeros((batch_size, num_steps), dtype=torch.long)
state = decoder.init_state(encoder(X), enc_valid_lens) # state = ((batch_size, num_steps, num_hiddens), hidden_state, enc_valid_lens)
output, state = decoder(X, state)
print(output.shape) # (batch_size, num_steps, vocab_size)
print(state[0].shape) # (batch_size, num_steps, num_hiddens)
print(state[1][0].shape) # (batch_size, num_hiddens)

data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
encoder = d2l.Seq2SeqEncoder(len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.005)
trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
d2l.plt.savefig(Path(__file__).stem+'_1.png', dpi=150, bbox_inches='tight') 
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
preds, _ = model.predict_step(
    data.build(engs, fras), d2l.try_gpu(), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p.tolist()):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
    _, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = attention_weights.reshape((1, 1, -1, data.num_steps))
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
d2l.plt.savefig(Path(__file__).stem+'_2.png', dpi=150, bbox_inches='tight') 