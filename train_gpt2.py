from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a 'bias' more of a mask but following the OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_embd)
        #calculate query, key, values for all heads in batch and move forward to tbe the batch
        #nh is "number of heads". hs is "head size", and C (number of channels) = nh*hs
        #e.g. in GPT2 (124M), n_head = 12, hs = 64, so nh*hs = C = 768 channels in the transformer
        qkv = self.c_attn(x)
        q, k ,c = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, nh, T, hs)
        #attention materializes the large (T, T) matrix for all the queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        ##output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="none") #Gaussian Error Linear Unit, supposed to have approximte='tanh' but not necessarily needed anymore
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            
            self.transformer = nn.ModuleDict(dict( #module to index into sub modules with keys like a dictionary
                wte = nn.Embedding(config.vocab_size, config.n_embd), #weights of the token embeddings
                wpe = nn.Embedding(config.block_size, config.n_embd), #weights of position embeddings, nn embedding fancy wrapper module over single array of numbers
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ##h is hiden instead of moduel dict is mudel list
                ln_f = nn.LayerNorm(config.n_embd) #final layer norm
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #final classifier, language model head