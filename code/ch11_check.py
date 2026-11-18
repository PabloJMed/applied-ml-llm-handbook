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
