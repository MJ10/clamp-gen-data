import math
import os
import torch
import torch.nn as nn
import pickle
import gzip

from clamp_common_eval.oracle import Oracle
from clamp_common_eval.datasets.placeholder import June_23_TwoWay_D1

class SmallTransformerOracle(Oracle):

    __comment__ = """
    Small transformer model trained to 92% accuracy on a D1 dataset
    """

    __data_split__ = June_23_TwoWay_D1

    def __init__(self):
        root = os.path.join(os.path.split(__file__)[0], '../../data/')
        self.tokenizer = pickle.load(gzip.open(root + '/tokenizer.pkl.gz', 'rb'))
        params = pickle.load(gzip.open(root + '/jun_23/small_transformer.pkl.gz', 'rb'))
        self.model = SmallTransformer(23, 1, 64, 4, 8, 60, 0.1)
        for a,b in zip(self.model.parameters(), params):
            a.data = torch.tensor(b)
        self.device = torch.device('cpu')
        self.eos_tok = 2 # TODO: make sure this matches the tokenizer's?

    def to(self, device):
        self.device = device
        self.model.to(device)

    def evaluate_many(self, sequences):
        """Returns sigmoid logits for each sequence"""
        x = self.tokenizer.process(sequences).to(self.device)
        return self.model(x.swapaxes(0,1), x.lt(self.eos_tok))

    def __call__(self, sequence):
        return self.evaluate_many([sequence])



class SmallTransformer(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid,
                 num_layers, num_head, max_len=60, dropout=0.1):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 1)
        self.embedding = nn.Embedding(num_tokens, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(num_hid, num_outputs)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[0, :] # This is weird... but this is what BERT calls pooling?
        # perhaps we could do like
        # pooled_x = (x * mask.unsqueeze(2)).sum(0) / mask.sum(1)
        # but with the mask dimensions right :P
        y = self.output(pooled_x)
        return y



# Taken from the PyTorch Transformer tutorial
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
