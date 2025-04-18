import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension

    def residual_std_factor(self) -> float:
        """
        we do not want std to explode in residual pathway, so we scale it down by a factor
        ```python
        x = normal(std=0.1)
        v = normal(std=0.1)
        # case 1
        x = x + v + .. (n times v)
        # std will become std=sqrt(0.1**2 + 0.1 ** 2 * n)
        # case 2
        x = x + sqrt(n) * (v + .. (n times v))
        # std will become std=sqrt(0.1**2 + (sqrt(n) * 0.1)**2 * n)
        ```
        """
        # we have 2 add operator in residual pathway per layer
        return (2 * self.n_layer) ** -0.5


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        nn.init.normal_(self.c_attn.weight, mean=0, std=0.02)
        nn.init.constant_(self.c_attn.bias, 0)
        # c_proj is used in residual pathway
        nn.init.normal_(
            self.c_proj.weight, mean=0, std=0.02 * config.residual_std_factor()
        )
        nn.init.constant_(self.c_proj.bias, 0)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        assert C == self.n_embd
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)  # [B, T, C] -> [B, T, 3 * C]
        q, k, v = qkv.split(C, dim=2)  # q, k, v: [B, T, C]
        k = k.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        nn.init.normal_(self.c_fc.weight, mean=0, std=0.02)
        nn.init.constant_(self.c_fc.bias, 0)
        # c_proj is used in residual pathway
        nn.init.normal_(
            self.c_proj.weight, mean=0, std=0.02 * config.residual_std_factor()
        )
        nn.init.constant_(self.c_proj.bias, 0)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        nn.init.constant_(self.ln_1.weight, 1)
        nn.init.constant_(self.ln_2.weight, 1)

    def forward(self, x):
        print(self.ln_1(x))
        # attn is an aggregation function, which will weighted sum input
        x = x + self.attn(self.ln_1(x))
        # mpl do operation for every single token individually
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        nn.init.normal_(self.transformer.wte.weight, mean=0, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, mean=0, std=0.01)
        nn.init.constant_(self.transformer.ln_f.weight, 1)

        # Share the same weight tensor between lm_head and wte
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            cross_entropy_result = logits.view(
                B * T, logits.size(2)
            )  # (B * T, vocab_size)
            loss = F.cross_entropy(cross_entropy_result, targets.view(-1))
        return logits, loss
