"""Chapter 26 · Interpretability: runnable checks (CPU, a few minutes; the control-probe accuracy varies slightly with thread count)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 26.6 an attention-only 2-layer transformer learns induction: [A B C ... A B C ...] with a random
# start and length, so the repeat can only be predicted by finding the previous occurrence
V, D, H, T = 32, 64, 2, 32

def batch(n):
    x = torch.randint(0, V, (n, T)); L = torch.randint(6, 13, (n,)); s = torch.randint(0, 4, (n,))
    mask = torch.zeros(n, T, dtype=torch.bool); lag = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        a, l = int(s[i]), int(L[i])
        x[i, a + l:a + 2 * l] = x[i, a:a + l]
        mask[i, a + l + 1:a + 2 * l] = True          # tokens predictable by induction
        lag[i] = l
    return x, mask, lag

class AttnOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D); self.pos = nn.Embedding(T, D)
        self.qkv = nn.ModuleList(nn.Linear(D, 3 * D, bias=False) for _ in range(2))
        self.out = nn.ModuleList(nn.Linear(D, D, bias=False) for _ in range(2))
        self.unemb = nn.Linear(D, V, bias=False)
        self.head_mask = torch.ones(2, H)                # set entries to 0 to ablate heads
        self.patterns = [None, None]
        self.resid = []                                   # residual stream after embedding, L0, L1 (for the logit lens)

    def forward(self, x):
        r = self.emb(x) + self.pos(torch.arange(T))      # the residual stream
        self.resid = [r]
        causal = torch.ones(T, T).tril().bool()
        for l in range(2):
            q, k, v = self.qkv[l](r).view(x.shape[0], T, 3, H, D // H).permute(2, 0, 3, 1, 4)
            att = (q @ k.transpose(-1, -2) / math.sqrt(D // H)).masked_fill(~causal, -1e9).softmax(-1)
            self.patterns[l] = att.detach()
            z = (att @ v) * self.head_mask[l][None, :, None, None]
            r = r + self.out[l](z.transpose(1, 2).reshape(x.shape[0], T, D))   # each head writes into the stream
            self.resid.append(r)
        return self.unemb(r)

model = AttnOnly()
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
for step in range(3000):
    x, _, _ = batch(64)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

def induction_acc(m, data=None):
    x, mask, _ = data if data is not None else batch(512)
    pred = m(x)[:, :-1].argmax(-1)
    return (pred == x[:, 1:])[mask[:, 1:]].float().mean().item()

print(f"accuracy on tokens predictable by induction: {induction_acc(model):.3f} (chance {1/V:.3f})")
x, mask, lag = batch(256); model(x)
for l in range(2):
    for h in range(H):
        a = model.patterns[l][:, h]
        prev = a[:, torch.arange(1, T), torch.arange(0, T - 1)].mean().item()
        ind = torch.stack([a[i, t, t - lag[i] + 1] for i in range(256) for t in range(T) if mask[i, t]]).mean().item()
        print(f"L{l}H{h}: previous-token score {prev:.2f}, induction score {ind:.2f}")
for heads in ([(0, 0)], [(0, 1)], [(1, 0)], [(1, 1)], [(1, 0), (1, 1)]):
    model.head_mask.fill_(1)
    for l, h in heads: model.head_mask[l, h] = 0
    print("ablate", " + ".join(f"L{l}H{h}" for l, h in heads), f"-> accuracy {induction_acc(model):.3f}")
model.head_mask.fill_(1)

# 26.5 activation patching: corrupt the first occurrence, then restore layer-1 head outputs from the clean run
def run_with_patch(x_corrupt, x_clean, heads):
    cache = {}
    def save(m, inp, out): cache["z"] = inp[0].detach()
    def patch(m, inp, out):
        z = inp[0].clone().view(x_corrupt.shape[0], T, H, D // H)
        zc = cache["z"].view(x_clean.shape[0], T, H, D // H)
        for h in heads: z[:, :, h] = zc[:, :, h]
        return F.linear(z.view(x_corrupt.shape[0], T, D), m.weight)
    hk = model.out[1].register_forward_hook(save); model(x_clean); hk.remove()
    hk = model.out[1].register_forward_hook(patch); logits = model(x_corrupt); hk.remove()
    return logits
x_clean, mask, lag = batch(512)
x_corrupt = x_clean.clone()
for i in range(512):                                     # replace the first occurrence with fresh random tokens
    a = int(mask[i].nonzero().min()) - 1 - int(lag[i])     # start of the first occurrence
    first = (torch.arange(T) >= a) & (torch.arange(T) < a + int(lag[i]))
    x_corrupt[i, first] = torch.randint(0, V, (int(first.sum()),))
def recovered(logits): return (logits[:, :-1].argmax(-1) == x_clean[:, 1:])[mask[:, 1:]].float().mean().item()
print(f"patching: clean {recovered(model(x_clean)):.3f}, corrupted {recovered(model(x_corrupt)):.3f}, "
      f"+ clean L1H0 {recovered(run_with_patch(x_corrupt, x_clean, [0])):.3f}, "
      f"+ clean L1H1 {recovered(run_with_patch(x_corrupt, x_clean, [1])):.3f}, "
      f"+ both {recovered(run_with_patch(x_corrupt, x_clean, [0, 1])):.3f}")

# ---------------------------------------------------------------------------
# 26.3 linear probes, with a control task
X = torch.randn(4000, 16)
hidden_net = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
latent = (X[:, 0] * X[:, 1] > 0).float()          # a non-linear feature of the input
reps = hidden_net(X).detach()
def probe_acc(features, labels):
    w = torch.zeros(features.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    o = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(400):
        l = F.binary_cross_entropy_with_logits(features[:3000] @ w + b, labels[:3000])
        o.zero_grad(); l.backward(); o.step()
    return ((features[3000:] @ w + b > 0).float() == labels[3000:]).float().mean().item()
trained = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
o = torch.optim.Adam(trained.parameters(), lr=1e-2)
for _ in range(600):
    l = F.binary_cross_entropy_with_logits(trained(X).squeeze(), latent); o.zero_grad(); l.backward(); o.step()
reps_trained = trained[1](trained[0](X)).detach()
control = torch.randint(0, 2, (4000,)).float()
print(f"probe for x0*x1>0: on raw input {probe_acc(X, latent):.3f}, random-network layer {probe_acc(reps, latent):.3f}, "
      f"trained-network layer {probe_acc(reps_trained, latent):.3f}; control (random labels) {probe_acc(reps_trained, control):.3f}")

# ---------------------------------------------------------------------------
# 26.7 superposition and a sparse autoencoder
n_feat, d_model = 24, 8
true_dirs = F.normalize(torch.randn(n_feat, d_model), dim=1)
def sample_acts(n, p=0.05):
    active = (torch.rand(n, n_feat) < p).float()
    return (active * torch.rand(n, n_feat)) @ true_dirs, active
acts, _ = sample_acts(50000)
print(f"{n_feat} sparse features stored in {d_model} dimensions; mean |cos| between feature directions "
      f"{(true_dirs @ true_dirs.T).abs().fill_diagonal_(0).sum().item() / (n_feat * (n_feat - 1)):.2f}")
def train_sae(m=32, l1=None, k=None, steps=8000):
    torch.manual_seed(1)
    enc, dec = nn.Linear(d_model, m), nn.Linear(m, d_model)
    with torch.no_grad():
        dec.weight.copy_(F.normalize(torch.randn(d_model, m), dim=0)); enc.weight.copy_(dec.weight.T)
    def code(x):
        f = F.relu(enc(x - dec.bias))
        if k is not None:                                # TopK SAE: keep the k largest latents
            v, i = f.topk(k, -1); f = torch.zeros_like(f).scatter(-1, i, v)
        return f
    o = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    for _ in range(steps):
        x = acts[torch.randint(0, len(acts), (512,))]
        f = code(x); loss = (dec(f) - x).pow(2).sum(-1).mean()
        if l1 is not None: loss = loss + l1 * (f * dec.weight.norm(dim=0)).sum(-1).mean()
        o.zero_grad(); loss.backward(); o.step()
        if k is not None:
            with torch.no_grad(): dec.weight.copy_(F.normalize(dec.weight, dim=0))
    with torch.no_grad():
        f = code(acts[:5000]); xh = dec(f)
        D_ = F.normalize(dec.weight.T, dim=1); best = (true_dirs @ D_.T).max(1).values
        fvu = ((xh - acts[:5000]).pow(2).sum() / (acts[:5000] - acts[:5000].mean(0)).pow(2).sum()).item()
    code.dec = dec                                       # keep the decoder for reconstruction checks
    return code, D_, (f > 0).float().sum(1).mean().item(), fvu, best

print(f"true active features per input: {(sample_acts(50000)[1].sum(1).mean()).item():.2f}")
saes = {}
for name, kw in (("L1 SAE, lambda 0.01", dict(l1=0.01)), ("L1 SAE, lambda 0.3", dict(l1=0.3)), ("TopK SAE, k=2", dict(k=2))):
    code, dec, l0, fvu, best = train_sae(**kw)
    saes[name] = code
    print(f"{name:20s}: L0 {l0:5.2f}, unexplained variance {fvu:.3f}, features recovered (cos > 0.9) "
          f"{(best > 0.9).sum().item()}/{n_feat}, median cos {best.median().item():.3f}")

# 26.8 steering: add a feature direction and read the change through the SAE
x0 = torch.zeros(1, d_model)
steered = x0 + 0.8 * true_dirs[3]
with torch.no_grad():
    f_steer = code(steered)
top = f_steer[0].argmax().item()
print(f"adding 0.8 x feature 3 to a zero activation: the TopK SAE's top latent has cos {(dec[top] @ true_dirs[3]).item():.2f} to feature 3")

# ---------------------------------------------------------------------------
# 26.2 composition scores: which layer-0 head's output does each layer-1 head read in its keys, queries, values?
# W_QK = W_Q^T W_K and W_OV = W_O W_V per head (D x D); K-composition = ||W_QK^(1) W_OV^(0)||_F / (||W_QK^(1)||_F ||W_OV^(0)||_F)
dh = D // H
def head_mats(l, h):
    W = model.qkv[l].weight.detach(); sl = slice(h * dh, (h + 1) * dh)
    Wq, Wk, Wv = W[:D][sl], W[D:2 * D][sl], W[2 * D:][sl]
    Wo = model.out[l].weight.detach()[:, sl]
    return Wq.T @ Wk, Wo @ Wv                              # W_QK, W_OV
def comp(A, B): return (torch.linalg.norm(A @ B) / (torch.linalg.norm(A) * torch.linalg.norm(B))).item()
g = torch.Generator().manual_seed(0)
rand = sum(comp(torch.randn(D, D, generator=g), torch.randn(D, D, generator=g)) for _ in range(200)) / 200
for h1 in range(H):
    WQK1, WOV1 = head_mats(1, h1)
    row = []
    for h0 in range(H):
        _, WOV0 = head_mats(0, h0)
        row.append(f"L0H{h0}: K {comp(WQK1, WOV0):.3f} Q {comp(WQK1.T, WOV0):.3f} V {comp(WOV1, WOV0):.3f}")
    print(f"composition into L1H{h1}: " + "; ".join(row) + f" (random baseline {rand:.3f})")

# 26.4 logit lens on the toy model: decode the residual stream after each layer with the unembedding
xl, ml, _ = batch(512)
with torch.no_grad():
    model(xl)
    for name, r in zip(("embeddings", "after layer 0", "after layer 1"), model.resid):
        pred = model.unemb(r)[:, :-1].argmax(-1)
        print(f"logit lens {name:14s}: accuracy on induction tokens {(pred == xl[:, 1:])[ml[:, 1:]].float().mean().item():.3f}")

# 26.5 attribution patching vs real patching, per layer-1 head, on a continuous metric (mean log-prob of the right token)
def metric(logits):
    lp = logits[:, :-1].log_softmax(-1).gather(-1, x_clean[:, 1:, None])[..., 0]
    return lp[mask[:, 1:]].mean()
store = {}
def grab(m, inp, out):
    store["z"] = inp[0]
    if inp[0].requires_grad: inp[0].retain_grad()
hk = model.out[1].register_forward_hook(grab)
with torch.no_grad(): model(x_clean)
z_clean = store["z"].detach().view(512, T, H, dh)
m_corr = metric(model(x_corrupt)); m_corr.backward()
z_corr, grad = store["z"].detach().view(512, T, H, dh), store["z"].grad.view(512, T, H, dh)
hk.remove(); model.zero_grad()
with torch.no_grad():
    m_clean = metric(model(x_clean)).item()
    for h in range(H):
        est = ((z_clean[:, :, h] - z_corr[:, :, h]) * grad[:, :, h]).sum().item()
        real = metric(run_with_patch(x_corrupt, x_clean, [h])).item() - m_corr.item()
        print(f"L1H{h}: patching changes the metric by {real:+.3f}, attribution patching estimates {est:+.3f} "
              f"(clean {m_clean:.3f}, corrupted {m_corr.item():.3f})")

# 26.7 L1 shrinkage: on inputs with exactly one active feature, how much of x does the reconstruction keep?
single = torch.zeros(2000, n_feat); idx = torch.randint(0, n_feat, (2000,))
single[torch.arange(2000), idx] = torch.rand(2000) * 0.5 + 0.5
x1 = single @ true_dirs
for name in ("L1 SAE, lambda 0.3", "TopK SAE, k=2"):
    c_ = saes[name]
    with torch.no_grad():
        xh = c_.dec(c_(x1))
    print(f"{name}: projection of reconstruction on input, <x_hat, x>/<x, x> = {((xh * x1).sum(1) / (x1 * x1).sum(1)).mean().item():.3f}")
print("1-D L1 solution: argmin (a - f)^2 + lambda f over f >= 0 is f = a - lambda/2; a = 1, lambda = 0.3 ->", 1 - 0.3 / 2)

# 26.2 a hand-checkable induction circuit: tokens A, B, C; dims 0-2 hold the current token, dims 3-5 the previous token
import numpy as np
X = np.zeros((4, 6))
for i, t in enumerate([0, 1, 2, 0]): X[i, t] = 1                  # sequence A B C A
W_OV0 = np.zeros((6, 6)); W_OV0[3:6, 0:3] = np.eye(3)               # previous-token head: copy token into "previous" dims
for i in range(1, 4): X[i] += W_OV0 @ X[i - 1]
W_QK1 = np.zeros((6, 6)); W_QK1[0:3, 3:6] = 5 * np.eye(3)           # induction head: my token vs your previous token
s = np.array([X[3] @ W_QK1 @ X[j] for j in range(4)]); a = np.exp(s) / np.exp(s).sum()
W_OV1 = np.zeros((6, 6)); W_OV1[0:3, 0:3] = np.eye(3)               # copy the attended token
print("toy induction: scores", s, "attention", a.round(3), "output on A,B,C", (sum(a[j] * W_OV1 @ X[j] for j in range(4))[:3]).round(3))
print("K-composition: W_QK1 @ W_OV0 is non-zero ->", np.abs(W_QK1 @ W_OV0).sum() > 0)

# 26.7 superposition capacity sketch: N random directions with all |cos| <= eps needs roughly N^2 exp(-d eps^2 / 2) < 1
for d_, eps in ((4096, 0.1), (4096, 0.2), (512, 0.2)):
    print(f"d={d_}, eps={eps}: N up to about exp(d eps^2 / 4) = {math.exp(d_ * eps ** 2 / 4):.3g}")

# 26.12 integrated gradients on F(x) = x1 * x2 at x = (2, 3), baseline 0
alphas = np.linspace(0, 1, 1001)
x_ = np.array([2.0, 3.0])
ig = x_ * np.array([np.trapezoid(alphas * x_[1], alphas), np.trapezoid(alphas * x_[0], alphas)])
print("IG", ig, "sum", ig.sum(), "= F(x) - F(0) =", x_[0] * x_[1], "; gradient x input", x_ * np.array([x_[1], x_[0]]))
for v in (0.721, 0.615, 0.974):
    print(f"normalised patching effect of {v}: {(v - 0.030) / (0.975 - 0.030):.3f}")
