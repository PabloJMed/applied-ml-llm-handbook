import math, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(0)

def attention_np(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    w = np.exp(scores - scores.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return w @ V, w
Q = np.random.randn(4, 8); K = np.random.randn(4, 8); V = np.random.randn(4, 8)
causal = np.tril(np.ones((4, 4), bool))
out, w = attention_np(Q, K, V, causal); print(np.round(w, 2)[1])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0):
        super().__init__()
        self.h, self.kv = n_heads, n_kv_heads or n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.k = nn.Linear(d_model, self.kv * self.dh, bias=False)
        self.v = nn.Linear(d_model, self.kv * self.dh, bias=False)
        self.o = nn.Linear(n_heads * self.dh, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2); v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        if self.kv != self.h:
            k = k.repeat_interleave(self.h // self.kv, dim=1); v = v.repeat_interleave(self.h // self.kv, dim=1)
        Tk = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        mask = torch.ones(T, Tk, dtype=torch.bool).tril(diagonal=Tk - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        att = F.dropout(F.softmax(scores, dim=-1), self.dropout, self.training)
        y = (att @ v).transpose(1, 2).reshape(B, T, self.h * self.dh)
        return self.o(y)

mha = MultiHeadAttention(32, 4, n_kv_heads=2)
x = torch.randn(2, 5, 32)
ref = F.scaled_dot_product_attention  # sanity: compare against torch for full heads
print(mha(x).shape)

def sinusoidal_pe(T, d):
    pos = torch.arange(T).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe
print(sinusoidal_pe(10, 16).shape)

def rope(x, base=10000.0):
    B, H, T, D = x.shape
    half = D // 2
    freqs = base ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = torch.arange(T, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
# relative property: <rope(q)_m, rope(k)_n> depends on m-n only
q = torch.randn(1, 1, 1, 8).repeat(1, 1, 6, 1); k = torch.randn(1, 1, 1, 8).repeat(1, 1, 6, 1)
rq, rk = rope(q), rope(k)
print(round((rq[0, 0, 3] @ rk[0, 0, 1]).item(), 4), round((rq[0, 0, 5] @ rk[0, 0, 3]).item(), 4))

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False); self.w3 = nn.Linear(d, hidden, bias=False); self.w2 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))
    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.ff(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, n_layers=4, max_len=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, idx, caches=None, start=0):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(start, start + T))
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        caches = [{} for _ in self.blocks]
        logits = self(idx, caches)
        out = idx
        for t in range(n_new):
            logits_last = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits_last, top_k); logits_last[logits_last < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits_last, -1), 1)
            out = torch.cat([out, nxt], 1)
            logits = self(nxt, caches, start=out.shape[1] - 1)
        return out

text = ("the courier picked up the order and the customer got the food hot. " * 40)
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])
model = MiniGPT(len(chars), d=64, n_heads=4, n_layers=2, max_len=64)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(300):
    ix = torch.randint(0, len(data) - 33, (32,))
    xb = torch.stack([data[i:i + 32] for i in ix]); yb = torch.stack([data[i + 1:i + 33] for i in ix])
    loss = F.cross_entropy(model(xb).reshape(-1, len(chars)), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
print("loss", round(loss.item(), 3))
model.eval()
# check KV cache gives same logits as full forward
seq = data[:20].unsqueeze(0)
full = model(seq)[:, -1]
caches = [{} for _ in model.blocks]
model(seq[:, :19], caches); inc = model(seq[:, 19:], caches, start=19)[:, -1]
print("cache match", torch.allclose(full, inc, atol=1e-5))
print("".join(itos[i] for i in model.generate(data[:10].unsqueeze(0), 40, top_k=5)[0].tolist()))
