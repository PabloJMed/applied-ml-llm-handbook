import torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np
torch.set_num_threads(1)   # the packed-LSTM training is much slower with many CPU threads
torch.manual_seed(0); random.seed(0); np.random.seed(0)

# toy corpus with two topics
food = "pizza burger sushi shawarma biryani pasta salad".split()
place = "dubai barcelona cairo riyadh doha kuwait amman".split()
sents = []
for _ in range(3000):
    if random.random() < .5:
        sents.append(["i", "ordered", random.choice(food), "and", random.choice(food), "tonight"])
    else:
        sents.append(["we", "flew", "to", random.choice(place), "from", random.choice(place)])
vocab = sorted({w for s in sents for w in s}); idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)

def skipgram_pairs(sents, window=2):
    pairs = []
    for s in sents:
        ids = [idx[w] for w in s]
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i: pairs.append((c, ids[j]))
    return torch.tensor(pairs)
pairs = skipgram_pairs(sents)
counts = torch.bincount(pairs[:, 1], minlength=V).float()
noise = counts ** 0.75; noise /= noise.sum()

class SGNS(nn.Module):
    def __init__(self, V, d=32):
        super().__init__()
        self.inp = nn.Embedding(V, d); self.out = nn.Embedding(V, d)
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d); nn.init.zeros_(self.out.weight)
    def forward(self, c, o, k=5):
        neg = torch.multinomial(noise, c.size(0) * k, replacement=True).view(-1, k)
        vc = self.inp(c)
        pos = F.logsigmoid((vc * self.out(o)).sum(-1))
        negs = F.logsigmoid(-(self.out(neg) @ vc.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + negs).mean()

m = SGNS(V); opt = torch.optim.Adam(m.parameters(), lr=0.01)
for ep in range(3):
    perm = torch.randperm(len(pairs))
    for b in range(0, len(pairs), 512):
        bt = pairs[perm[b:b + 512]]
        loss = m(bt[:, 0], bt[:, 1]); opt.zero_grad(); loss.backward(); opt.step()
E = F.normalize(m.inp.weight.detach(), dim=1)
def nearest(w, k=4):
    s = E @ E[idx[w]]; return [vocab[i] for i in s.argsort(descending=True)[1:k + 1].tolist()]
print("sushi ->", nearest("sushi")); print("dubai ->", nearest("dubai"))

# LSTM classifier with packing
class LSTMClassifier(nn.Module):
    def __init__(self, V, d=32, h=64, n_cls=2):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * h, n_cls)
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)          # h_n: (2, B, h)
        return self.head(torch.cat([h_n[0], h_n[1]], dim=-1))

# task: does the sequence contain token 5 before token 7 ? (order-sensitive)
def make(n=4000, Vt=20, L=12):
    X = torch.zeros(n, L, dtype=torch.long); y = torch.zeros(n, dtype=torch.long); lens = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        l = random.randint(4, L); seq = [random.randint(1, Vt - 1) for _ in range(l)]
        a, b = random.sample(range(l), 2); seq[a], seq[b] = 5, 7
        y[i] = int(a < b); X[i, :l] = torch.tensor(seq); lens[i] = l
    return X, y, lens
X, y, lens = make()
clf = LSTMClassifier(20); opt = torch.optim.Adam(clf.parameters(), lr=3e-3)
for ep in range(8):
    perm = torch.randperm(3200)
    for b in range(0, 3200, 64):
        ii = perm[b:b + 64]
        loss = F.cross_entropy(clf(X[ii], lens[ii]), y[ii]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    acc = (clf(X[3200:], lens[3200:]).argmax(-1) == y[3200:]).float().mean().item()
print("lstm order acc", round(acc, 3))
# bag of embeddings baseline
bag = nn.Sequential(nn.EmbeddingBag(20, 32, mode="mean"), nn.Linear(32, 2))
opt = torch.optim.Adam(bag.parameters(), lr=3e-3)
for ep in range(8):
    for b in range(0, 3200, 64):
        xb = X[b:b+64]; loss = F.cross_entropy(bag[1](bag[0](xb, per_sample_weights=None) if False else bag[0](xb)), y[b:b+64]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    acc2 = (bag[1](bag[0](X[3200:])).argmax(-1) == y[3200:]).float().mean().item()
print("bag acc", round(acc2, 3))

# Bahdanau additive attention
class AdditiveAttention(nn.Module):
    def __init__(self, d_dec, d_enc, d_att=64):
        super().__init__()
        self.W_q = nn.Linear(d_dec, d_att, bias=False); self.W_k = nn.Linear(d_enc, d_att, bias=False)
        self.v = nn.Linear(d_att, 1, bias=False)
    def forward(self, s, H, mask):           # s: (B, d_dec), H: (B, T, d_enc), mask: (B, T) True = real token
        scores = self.v(torch.tanh(self.W_q(s).unsqueeze(1) + self.W_k(H))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = scores.softmax(-1)
        return (alpha.unsqueeze(-1) * H).sum(1), alpha
att = AdditiveAttention(64, 128)
s = torch.randn(3, 64); H = torch.randn(3, 7, 128); mask = torch.ones(3, 7, dtype=torch.bool); mask[0, 5:] = False
ctx, a = att(s, H, mask); print(ctx.shape, a.shape, a[0, 5:].tolist(), round(a.sum(-1)[0].item(), 4))

# ---- review additions (fast; no training) --------------------------------------------------
import math
sig = lambda x: 1 / (1 + math.exp(-x))
# 8.2 SGNS loss for one positive (dot 2.0) and two negatives (dots 0.5 and -1.0)
terms = [-math.log(sig(2.0)), -math.log(sig(-0.5)), -math.log(sig(1.0))]
print("SGNS terms", [round(t, 3) for t in terms], "loss", round(sum(terms), 3))
print("grad coefficients on v_c: positive", round(-(1 - sig(2.0)), 4), " negatives", round(sig(0.5), 4), round(sig(-1.0), 4))
print("full softmax vs SGNS cost per pair (V=50k, k=5):", 50000 / 6)
# check the analytic gradient against autograd
vc = torch.randn(8, requires_grad=True); uo, u1, u2 = torch.randn(8), torch.randn(8), torch.randn(8)
L = -(F.logsigmoid(uo @ vc) + F.logsigmoid(-(u1 @ vc)) + F.logsigmoid(-(u2 @ vc))); L.backward()
g_hand = -(1 - torch.sigmoid(uo @ vc)) * uo + torch.sigmoid(u1 @ vc) * u1 + torch.sigmoid(u2 @ vc) * u2
print("SGNS hand gradient matches autograd:", torch.allclose(vc.grad, g_hand.detach(), atol=1e-6))
# weight tying saving for a 128,256 x 2,048 table
print("tied embedding params (M):", round(128256 * 2048 / 1e6, 1))
# 1/sqrt(d): variance of a dot product of two random unit-variance vectors
q, k = torch.randn(100000, 64), torch.randn(100000, 64)
print("dot-product std at d=64:", round((q * k).sum(-1).std().item(), 2), " after /sqrt(d):", round(((q * k).sum(-1) / 8).std().item(), 3))
# 8.4 per-step decay implied by the figure's numbers, and forget-gate products over 20 steps
print("per-step factor RNN", round(4.6e-8 ** (1 / 20), 3), " LSTM", round(1.1e-2 ** (1 / 20), 3), " ratio", round(1.1e-2 / 4.6e-8))
for f in (0.5, sig(1.0), 0.9):
    print(f"forget gate {f:.3f}: product over 20 steps {f ** 20:.2e}")
lstm = nn.LSTM(256, 256); gru = nn.GRU(256, 256)
print("params LSTM", sum(p.numel() for p in lstm.parameters()), " GRU", sum(p.numel() for p in gru.parameters()),
      " ratio", round(sum(p.numel() for p in gru.parameters()) / sum(p.numel() for p in lstm.parameters()), 3))
print("PyTorch default LSTM forget bias is not 1:", float(nn.LSTM(4, 4).bias_ih_l0[4:8].detach().abs().max()) < 1)
# 8.5 exposure bias: chance of an error-free 100-token output at per-token error 1%
print("P(no error in 100 tokens at 1%):", round(0.99 ** 100, 3))
# Q6: standard error of a 10% hit rate on 50,000 held-out sessions
print("SE of hit rate 0.10 on 50k sessions (points):", round(100 * math.sqrt(0.1 * 0.9 / 50000), 2))
