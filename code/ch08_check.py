import torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np
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
