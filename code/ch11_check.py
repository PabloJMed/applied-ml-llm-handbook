import torch, torch.nn as nn, torch.nn.functional as F, math
torch.manual_seed(0)
# Chinchilla arithmetic
def chinchilla(C):
    N = math.sqrt(C / (6 * 20)); D = 20 * N; return N, D
for C in [1e21, 1e23, 1e24]:
    N, D = chinchilla(C); print(f"C={C:.0e}: N={N/1e9:.1f}B params, D={D/1e9:.0f}B tokens")
# Chinchilla loss fit (published constants, approx)
def L(N, D, E=1.69, A=406.4, B=410.7, a=0.34, b=0.28): return E + A / N**a + B / D**b
print(round(L(70e9, 1.4e12), 3), round(L(280e9, 300e9), 3))

# top-k MoE layer
class MoE(nn.Module):
    def __init__(self, d, n_exp=8, k=2, d_ff=64):
        super().__init__()
        self.router = nn.Linear(d, n_exp, bias=False); self.k = k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d)) for _ in range(n_exp)])
    def forward(self, x):                       # x: (T, d) tokens
        logits = self.router(x)                 # (T, E)
        w, idx = logits.topk(self.k, dim=-1)    # (T, k)
        w = w.softmax(-1)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel(): out[tok] += w[tok, slot, None] * expert(x[tok])
        probs = logits.softmax(-1)              # load-balancing aux loss (Switch style)
        frac = F.one_hot(idx[:, 0], len(self.experts)).float().mean(0)
        aux = len(self.experts) * (frac * probs.mean(0)).sum()
        return out, aux
moe = MoE(16); y, aux = moe(torch.randn(32, 16)); print(y.shape, round(aux.item(), 3))
tot = sum(p.numel() for p in moe.parameters()); act = sum(p.numel() for p in moe.experts[0].parameters()) * 2 + moe.router.weight.numel()
print("total", tot, "active per token", act)

# CLIP-style symmetric InfoNCE
def clip_loss(img, txt, t=0.07):
    img, txt = F.normalize(img, dim=-1), F.normalize(txt, dim=-1)
    logits = img @ txt.T / t; labels = torch.arange(len(img))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
a = torch.randn(8, 32); print(round(clip_loss(a, a + 0.1 * torch.randn(8, 32)).item(), 4), round(clip_loss(a, torch.randn(8, 32)).item(), 3))

# mean pooling with mask
def mean_pool(h, mask):
    m = mask.unsqueeze(-1).float(); return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
h = torch.randn(2, 5, 4); mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,1]])
print(torch.allclose(mean_pool(h, mask)[0], h[0, :3].mean(0)))
