"""Chapter 7 · Optimisation and training dynamics: runnable checks (CPU, a few seconds)."""
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# 7.1 AdamW from scratch matches torch.optim.AdamW
def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    p = p * (1 - lr * wd)                      # decoupled weight decay
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return p - lr * m_hat / (v_hat.sqrt() + eps), m, v

w = torch.randn(5, requires_grad=True)
opt = torch.optim.AdamW([w], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
p, m, v = w.detach().clone(), torch.zeros(5), torch.zeros(5)
for t in range(1, 4):
    g = torch.randn(5)
    w.grad = g.clone(); opt.step()
    p, m, v = adamw_step(p, g, m, v, t)
assert torch.allclose(p, w.detach(), atol=1e-7)
print("AdamW from scratch matches torch:", True)
# first step moves every coordinate by ~lr regardless of gradient scale
p0 = torch.tensor([1.0, 1.0]); g0 = torch.tensor([1e-4, 1e2])
p1, _, _ = adamw_step(p0, g0, torch.zeros(2), torch.zeros(2), 1, wd=0.0)
print("first-step update for grads 1e-4 and 1e2:", [round(x, 6) for x in (p0 - p1).tolist()])

# 7.2 learning-rate schedules
def cosine(step, total, peak=3e-4, warmup=100, final=0.1):
    if step < warmup: return peak * step / warmup
    prog = (step - warmup) / (total - warmup)
    return peak * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * prog)))

def wsd(step, total, peak=3e-4, warmup=100, decay_frac=0.2, final=0.0):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup: return peak * step / warmup
    if step < decay_start: return peak
    prog = (step - decay_start) / (total - decay_start)
    return peak * (1 - prog) + final * prog

total = 1000
for s in (50, 500, 800, 900, 999):
    print(f"step {s}: cosine {cosine(s, total):.2e}  wsd {wsd(s, total):.2e}")

# 7.3 gradient noise scale on a logistic regression problem
N, D = 20000, 20
X = torch.randn(N, D); w_true = torch.randn(D)
y = (torch.sigmoid(X @ w_true) > torch.rand(N)).float()
wq = torch.zeros(D)
def per_example_grads(wq, idx):
    p = torch.sigmoid(X[idx] @ wq)
    return (p - y[idx])[:, None] * X[idx]          # (n, D)
G = per_example_grads(wq, torch.arange(N))
g_mean = G.mean(0)
trace_cov = G.var(0, unbiased=True).sum()
b_simple = (trace_cov / g_mean.pow(2).sum()).item()
print(f"B_simple at init: {b_simple:.1f} examples")
# later in training the gradient shrinks and the noise scale grows
for _ in range(300):
    wq -= 0.5 * per_example_grads(wq, torch.arange(N)).mean(0)
G = per_example_grads(wq, torch.arange(N))
b_late = (G.var(0, unbiased=True).sum() / G.mean(0).pow(2).sum()).item()
print(f"B_simple after 300 full-batch steps: {b_late:.0f} examples")

# 7.4 Muon's Newton-Schulz orthogonalisation
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transpose = X.shape[0] > X.shape[1]
    if transpose: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transpose else X

Gm = torch.randn(64, 32) @ torch.diag(torch.logspace(0, -3, 32))   # badly conditioned "gradient"
s_before = torch.linalg.svdvals(Gm)
s_after = torch.linalg.svdvals(newton_schulz(Gm))
print(f"singular values before: max/min = {(s_before.max() / s_before.min()).item():.0f}")
print(f"after Newton-Schulz: range {s_after.min().item():.2f}-{s_after.max().item():.2f}")

# 7.5 why muP: how much one Adam step changes the output, as width grows
def output_change(width, lr, mup):
    torch.manual_seed(1)
    x = torch.randn(256, 64)
    l1 = nn.Linear(64, width, bias=False); l2 = nn.Linear(width, width, bias=False)
    l3 = nn.Linear(width, 1, bias=False)
    nn.init.normal_(l1.weight, std=1 / math.sqrt(64)); nn.init.normal_(l2.weight, std=1 / math.sqrt(width))
    nn.init.normal_(l3.weight, std=(1 / width) if mup else (1 / math.sqrt(width)))
    net = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)
    groups = [{"params": l1.parameters(), "lr": lr},
              {"params": list(l2.parameters()) + list(l3.parameters()), "lr": lr * (64 / width if mup else 1)}]
    opt = torch.optim.Adam(groups)
    out0 = net(x).detach()
    loss = (net(x) - 1).pow(2).mean(); loss.backward(); opt.step()
    return (net(x).detach() - out0).abs().mean().item()

for width in (64, 256, 1024):
    print(f"width {width}: standard {output_change(width, 1e-3, False):.3f}  muP {output_change(width, 1e-3, True):.3f}")

# 7.6 precision
one = torch.tensor(1.0)
print("1 + 1e-3 in bf16:", (one.bfloat16() + torch.tensor(1e-3).bfloat16()).item(),
      " in fp32:", (one + 1e-3).item())
print("bf16 eps:", torch.finfo(torch.bfloat16).eps, " fp16 max:", torch.finfo(torch.float16).max,
      " bf16 max: %.2e" % torch.finfo(torch.bfloat16).max)
if hasattr(torch, "float8_e4m3fn"):
    print("fp8 e4m3 max:", torch.finfo(torch.float8_e4m3fn).max, " e5m2 max:", torch.finfo(torch.float8_e5m2).max)
    vals = torch.tensor([0.1, 1.0, 3.3, 300.0, 1000.0])
    print("round-trip e4m3:", vals.to(torch.float8_e4m3fn).float().tolist())
# per-tensor scaling keeps small values representable
small = torch.tensor([1e-4, 3e-4, 5e-4])
scale = 448 / small.abs().max()
print("unscaled e4m3:", small.to(torch.float8_e4m3fn).float().tolist(),
      " scaled:", [round(x, 6) for x in ((small * scale).to(torch.float8_e4m3fn).float() / scale).tolist()])

# 7.7 z-loss and gradient clipping
logits = torch.tensor([[12.0, 10.0, 9.0], [2.0, 1.0, 0.5]])
log_z = torch.logsumexp(logits, dim=-1)
print("log Z per row:", [round(x, 3) for x in log_z.tolist()], " z-loss (1e-4 * mean log Z^2):",
      round((1e-4 * log_z.pow(2)).mean().item(), 5))
params = [torch.zeros(3, requires_grad=True), torch.zeros(4, requires_grad=True)]
params[0].grad = torch.tensor([3.0, 0.0, 0.0]); params[1].grad = torch.tensor([0.0, 4.0, 0.0, 0.0])
norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
print("grad norm before clip:", norm.item(), " after:", round(torch.cat([p.grad for p in params]).norm().item(), 4))
