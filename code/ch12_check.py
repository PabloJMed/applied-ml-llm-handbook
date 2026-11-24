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

# ---- review additions ----------------------------------------------------------------------
import numpy as np

# 12.5 three answers for a 1e23-FLOP budget
E3, A3, B3, a3, b3 = 1.69, 406.4, 410.7, 0.34, 0.28           # Hoffmann et al., approach 3
EB, AB, BB, aB, bB = 1.8172, 482.01, 2085.43, 0.3478, 0.3658   # Besiroglu et al. (2024) re-fit
def n_opt(C, A, B, a, b):
    G = (a * A / (b * B)) ** (1 / (a + b))
    return G * (C / 6) ** (b / (a + b))
C = 1e23
for name, N in [("20 tokens/param", math.sqrt(C / 120)),
                ("Besiroglu re-fit", n_opt(C, AB, BB, aB, bB)),
                ("approach-3 fit", n_opt(C, A3, B3, a3, b3))]:
    D = C / (6 * N)
    print(f"1e23 FLOPs, {name:17s}: N = {N/1e9:5.1f}B, D = {D/1e9:5.0f}B tokens, {D/N:4.0f} tokens/param")

# Q6: same target loss, plus 1e12 inference tokens a month for 12 months (2N FLOPs per token)
LB = lambda N, D: EB + AB / N**aB + BB / D**bB
N0 = n_opt(C, AB, BB, aB, bB); target = LB(N0, C / (6 * N0))
def total_cost(N, inf_tokens=12e12):
    D = (BB / (target - EB - AB / N**aB)) ** (1 / bB)          # tokens needed to reach the target loss
    return 6 * N * D, 2 * N * inf_tokens, D
grid = np.linspace(6e9, 30e9, 241)
best = min(grid, key=lambda N: sum(total_cost(N)[:2]))
for N in (N0, 13e9, best, 8e9):
    tr, inf, D = total_cost(N)
    print(f"N = {N/1e9:5.1f}B: D = {D/1e12:5.2f}T, train {tr:.2e}, inference {inf:.2e}, total {tr+inf:.2e} FLOPs")

# 12.2 index memory for 1M documents in float32
for dim in (768, 4096):
    print(f"1M vectors x {dim} dims x 4 bytes = {1e6*dim*4/1e9:.1f} GB")

# 12.6 decode roofline for an 8B model in bf16 on one GPU with 3.35 TB/s of HBM bandwidth
weights = 16.06e9; bw = 3.35e12; batch = 64; price = 2.50                 # $ per GPU-hour (assumed)
kv = 2 * 32 * 8 * 128 * 2 * 2000 * batch                                   # GQA-8 cache, 2k tokens each
for label, bytes_ in (("weights only", weights), ("weights + 2k-token KV", weights + kv)):
    step = bytes_ / bw; tps = batch / step
    print(f"{label:22s}: step {step*1e3:4.1f} ms, {tps:6.0f} tokens/s, ${price/3600/tps*1e6:.3f} per 1M output tokens")
print(f"compute-bound batch on a 989-TFLOP/s, 3.35-TB/s GPU: about {989e12/bw:.0f} sequences")

# 12.6 cascade: cost relative to always using the large model
c_small, c_large = 0.05, 1.0
for p in (0.1, 0.2, 0.5, 0.95):
    print(f"escalate {p:.0%}: cost {c_small + p*c_large:.2f} of large-only")

# 12.9 landscape arithmetic: active share and memory
for name, total, active in (("DeepSeek-V3", 671e9, 37e9), ("Kimi K2", 1000e9, 32e9),
                            ("Qwen3-235B-A22B", 235e9, 22e9), ("gpt-oss-120b", 117e9, 5.1e9),
                            ("Llama 4 Maverick", 400e9, 17e9)):
    print(f"{name:17s}: active {active/total:5.1%}, weights {total/1e9:5.0f} GB at 8 bits, {2*total/1e9:5.0f} GB in bf16")
print(f"gpt-oss-120b at ~4.25 bits per parameter: {117e9*4.25/8/1e9:.0f} GB")

# Research exercise: smooth per-token accuracy looks 'emergent' under exact match
logN = np.linspace(8, 12, 9)                          # 1e8 .. 1e12 parameters
p_tok = 1 / (1 + np.exp(-(logN - 9.5) * 1.2))         # smooth per-token accuracy
for n, p in zip(logN, p_tok):
    print(f"N=1e{n:.1f}: per-token {p:.3f}, exact match on 10 tokens {p**10:.3f}")
