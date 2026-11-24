"""Chapter 11 · Architectures beyond the vanilla transformer: runnable checks (CPU, seconds)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 11.2 top-k MoE layer with the Switch/GShard load-balancing loss
class MoE(nn.Module):
    def __init__(self, d, n_experts=8, k=2, hidden=64):
        super().__init__()
        self.k, self.n = k, n_experts
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, d))
                                     for _ in range(n_experts))

    def forward(self, x):                          # x: (tokens, d)
        probs = F.softmax(self.router(x), dim=-1)  # (tokens, n)
        topv, topi = probs.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)   # renormalise the chosen gates
        out = torch.zeros_like(x)
        for e in range(self.n):
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if rows.numel():
                out[rows] += topv[rows, slot, None] * self.experts[e](x[rows])
        # f_e: fraction of routed slots that went to expert e; P_e: mean router probability
        f = torch.bincount(topi.flatten(), minlength=self.n).float() / topi.numel()
        P = probs.mean(0)
        aux = self.n * (f * P).sum()
        return out, aux, f

moe = MoE(16)
x = torch.randn(512, 16)
y, aux, f = moe(x)
print("output", tuple(y.shape), " aux loss %.3f" % aux.item(), " load per expert", [round(v, 3) for v in f.tolist()])
# aux = n * sum f*P is 1.0 when routing is perfectly uniform and grows with imbalance
n = 8
uni = torch.full((n,), 1 / n)
skew = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0])
print("aux uniform %.2f  skewed %.2f" % (n * (uni * uni).sum().item(), n * (skew * skew).sum().item()))

# 11.2 auxiliary-loss-free balancing (DeepSeek-V3 style): a per-expert bias used only for selection
scores = torch.randn(4096, n) + torch.tensor([1.5, 1.0, 0.5, 0, 0, 0, -0.5, -1.0])  # biased router
bias = torch.zeros(n)
for step in range(200):
    topi = (scores + bias).topk(2, dim=-1).indices
    load = torch.bincount(topi.flatten(), minlength=n).float() / topi.numel()
    bias -= 0.01 * torch.sign(load - 1 / n)          # push overloaded experts down, underloaded up
    if step in (0, 199):
        print(f"step {step}: max/min expert load {load.max().item():.3f}/{load.min().item():.3f}")

# 11.2 total vs active parameters for a fine-grained MoE FFN
d_model, d_expert, n_routed, n_shared, k = 4096, 1024, 128, 1, 8
per_expert = 3 * d_model * d_expert                  # SwiGLU: gate, up, down
total = (n_routed + n_shared) * per_expert
active = (k + n_shared) * per_expert
dense_ffn = 3 * d_model * (4 * d_model * 2 // 3)     # a dense SwiGLU FFN of the usual size
print(f"MoE FFN per layer: total {total/1e6:.0f}M, active {active/1e6:.0f}M; dense FFN {dense_ffn/1e6:.0f}M")

# 11.3 KV-cache bytes per token per layer (bf16): MHA vs GQA vs MLA
def kv_bytes(n_heads, head_dim, kv_heads=None, latent=None, rope_dim=0):
    if latent is not None:
        return 2 * (latent + rope_dim)
    return 2 * 2 * (kv_heads or n_heads) * head_dim
mha = kv_bytes(128, 128); gqa = kv_bytes(128, 128, kv_heads=8); mla = kv_bytes(128, 128, latent=512, rope_dim=64)
print(f"KV bytes/token/layer: MHA {mha}, GQA-8 {gqa}, MLA {mla}  (MHA/MLA = {mha/mla:.0f}x)")
print(f"128k-token context, 61 layers: MHA {mha*61*131072/1e9:.0f} GB, MLA {mla*61*131072/1e9:.1f} GB")

# 11.4 causal linear attention: parallel form == recurrent form
T, d = 6, 4
q, k_, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
phi = lambda z: F.elu(z) + 1                          # positive feature map
mask = torch.tril(torch.ones(T, T))
A = (phi(q) @ phi(k_).T) * mask
parallel = (A @ v) / A.sum(-1, keepdim=True)
S = torch.zeros(d, d); z = torch.zeros(d); rec = []
for t in range(T):
    S = S + torch.outer(phi(k_[t]), v[t]); z = z + phi(k_[t])
    rec.append((phi(q[t]) @ S) / (phi(q[t]) @ z))
print("linear attention parallel == recurrent:", torch.allclose(parallel, torch.stack(rec), atol=1e-5))

# 11.4 gated delta rule (DeltaNet with a forget gate): the state overwrites, it does not only add
def gated_delta(q, k, v, beta, alpha):
    S = torch.zeros(k.shape[1], v.shape[1]); out = []
    for t in range(len(q)):
        kt = F.normalize(k[t], dim=0)
        S = alpha[t] * S                                       # forget (decay)
        S = S + beta[t] * torch.outer(kt, v[t] - kt @ S)       # delta-rule correction
        out.append(q[t] @ S)
    return torch.stack(out)
# associative recall: write key->value pairs, then query a key that was written twice
keys = torch.eye(4); vals = torch.eye(4) * 3
seq_k = torch.stack([keys[0], keys[1], keys[0]]); seq_v = torch.stack([vals[2], vals[3], vals[1]])
qry = torch.stack([keys[1], keys[1], keys[0]])
plain = torch.cumsum(torch.einsum("ti,tj->tij", seq_k, seq_v), 0)   # additive linear attention state
print("additive state, query key0 ->", (qry[-1] @ plain[-1]).tolist())
print("delta rule,     query key0 ->", gated_delta(qry, seq_k, seq_v, torch.ones(3), torch.ones(3))[-1].tolist())

# 11.4 memory per sequence: KV cache grows with T, a recurrent state does not
layers, kv_heads, head_dim = 32, 8, 128
for T_ in (4_096, 131_072, 1_048_576):
    kv = 2 * 2 * layers * kv_heads * head_dim * T_ / 1e9
    print(f"T={T_:>9,}: attention KV cache {kv:6.2f} GB   recurrent state (32 layers, 32 heads, 128x128) "
          f"{2*layers*32*128*128/1e9:.2f} GB")

# 11.7 masked-diffusion training step: mask a random fraction t of tokens, predict them, weight by 1/t
V, L = 50, 12
tokens = torch.randint(1, V, (4, L)); MASK = 0
t = torch.rand(4, 1).clamp(min=0.05)
masked = torch.rand(4, L) < t
inp = tokens.masked_fill(masked, MASK)
logits = torch.randn(4, L, V)                         # stand-in for a bidirectional transformer
ce = F.cross_entropy(logits.transpose(1, 2), tokens, reduction="none")
loss = ((ce * masked) / t).sum() / (4 * L)
print("masked fraction per sequence:", [round(v, 2) for v in masked.float().mean(1).tolist()], " loss %.2f" % loss.item())

# ---- review additions ----------------------------------------------------------------------
import math
import numpy as np

# 11.2 capacity factor: share of routed tokens dropped for the skewed loads above (step 0) and balanced loads
scores = torch.randn(4096, n) + torch.tensor([1.5, 1.0, 0.5, 0, 0, 0, -0.5, -1.0])
load0 = torch.bincount(scores.topk(2, dim=-1).indices.flatten(), minlength=n).float() / (4096 * 2)
for cf in (1.0, 1.25, 2.0):
    drop = torch.clamp(load0 - cf / n, min=0).sum().item()
    print(f"capacity factor {cf}: skewed router drops {drop:.1%} of routed tokens")
print("skewed loads:", [round(v, 3) for v in sorted(load0.tolist(), reverse=True)])

# 11.2 experts touched in one decode step: 128 routed experts, top 8, B tokens routed independently
for B_ in (1, 4, 16, 64, 256):
    print(f"batch {B_:>3}: expected experts touched {128 * (1 - (1 - 8/128) ** B_):6.1f} of 128")

# 11.2 router z-loss example
logits_ex = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
print("router z-loss for one token:", round((torch.logsumexp(logits_ex, -1) ** 2).mean().item(), 3))

# 11.4 1-D selective SSM: zero-order-hold discretisation with A = -1, B = C = 1
def ssm_1d(x, deltas, A=-1.0):
    h, hs = 0.0, []
    for xt, dt in zip(x, deltas):
        a_bar = math.exp(dt * A); b_bar = (a_bar - 1) / A        # ZOH: (exp(dA) - 1)/A * B
        h = a_bar * h + b_bar * xt; hs.append(round(h, 3))
    return hs
x = [2.0, 0.3, -0.5, 0.4]
print("fixed delta 0.5:      ", ssm_1d(x, [0.5] * 4), " a_bar", round(math.exp(-0.5), 3), "b_bar", round(1 - math.exp(-0.5), 3))
print("selective (3, .01x3): ", ssm_1d(x, [3.0, 0.01, 0.01, 0.01]), " a_bar(3)", round(math.exp(-3), 3))

# 11.4 the recurrence h_t = a_t h_{t-1} + b_t is an associative scan
def combine(p, q):            # apply p then q
    return (p[0] * q[0], q[0] * p[1] + q[1])
rng = np.random.default_rng(0)
a_, b_ = rng.uniform(0.5, 1, 8), rng.standard_normal(8)
h, seq = 0.0, []
for at, bt in zip(a_, b_):
    h = at * h + bt; seq.append(h)
acc = (a_[0], b_[0]); left = [(a_[i], b_[i]) for i in range(1, 4)]; right = [(a_[i], b_[i]) for i in range(4, 8)]
L_ = acc
for e in left: L_ = combine(L_, e)
R_ = right[0]
for e in right[1:]: R_ = combine(R_, e)
print("scan: two halves combined == sequential:", np.isclose(combine(L_, R_)[1], seq[-1]))

# 11.4 state-space duality: scalar-decay SSM == masked 'attention' y = (L o C B^T) x
T_, N_ = 6, 4
a_t = rng.uniform(0.6, 1.0, T_); Bm = rng.standard_normal((T_, N_)); Cm = rng.standard_normal((T_, N_)); xs = rng.standard_normal(T_)
h = np.zeros(N_); y_rec = []
for t in range(T_):
    h = a_t[t] * h + Bm[t] * xs[t]; y_rec.append(Cm[t] @ h)
Lmask = np.array([[np.prod(a_t[s + 1:t + 1]) if s <= t else 0.0 for s in range(T_)] for t in range(T_)])
y_mat = (Lmask * (Cm @ Bm.T)) @ xs
print("SSD: recurrent == (L o C B^T) x:", np.allclose(y_rec, y_mat))

# 11.4 the delta rule is one gradient step on 0.5 * ||S^T k - v||^2
S0 = torch.randn(4, 4); kk = F.normalize(torch.randn(4), dim=0); vv = torch.randn(4)
S_ = S0.clone().requires_grad_(True)
(0.5 * ((S_.T @ kk - vv) ** 2).sum()).backward()
beta = 1.0
print("delta rule == gradient step:", torch.allclose(S0 - beta * S_.grad, S0 + beta * torch.outer(kk, vv - S0.T @ kk), atol=1e-6),
      "| after the step S^T k == v:", torch.allclose((S0 + torch.outer(kk, vv - S0.T @ kk)).T @ kk, vv, atol=1e-5))

# 11.6 MTP as a draft: one extra token accepted with probability 0.85-0.90
for acc_ in (0.85, 0.90):
    print(f"acceptance {acc_:.2f}: tokens per forward pass {1 + acc_:.2f}")

# 11.9 MoE active share for the fine-grained layer above
print(f"active share {active/total:.1%}")
