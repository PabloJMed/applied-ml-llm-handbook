"""Chapter 26 · Interpretability: runnable checks (CPU, about a minute)."""
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

    def forward(self, x):
        r = self.emb(x) + self.pos(torch.arange(T))      # the residual stream
        causal = torch.ones(T, T).tril().bool()
        for l in range(2):
            q, k, v = self.qkv[l](r).view(x.shape[0], T, 3, H, D // H).permute(2, 0, 3, 1, 4)
            att = (q @ k.transpose(-1, -2) / math.sqrt(D // H)).masked_fill(~causal, -1e9).softmax(-1)
            self.patterns[l] = att.detach()
            z = (att @ v) * self.head_mask[l][None, :, None, None]
            r = r + self.out[l](z.transpose(1, 2).reshape(x.shape[0], T, D))   # each head writes into the stream
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
    return code, D_, (f > 0).float().sum(1).mean().item(), fvu, best

print(f"true active features per input: {(sample_acts(50000)[1].sum(1).mean()).item():.2f}")
for name, kw in (("L1 SAE, lambda 0.01", dict(l1=0.01)), ("L1 SAE, lambda 0.3", dict(l1=0.3)), ("TopK SAE, k=2", dict(k=2))):
    code, dec, l0, fvu, best = train_sae(**kw)
    print(f"{name:20s}: L0 {l0:5.2f}, unexplained variance {fvu:.3f}, features recovered (cos > 0.9) "
          f"{(best > 0.9).sum().item()}/{n_feat}, median cos {best.median().item():.3f}")

# 26.8 steering: add a feature direction and read the change through the SAE
x0 = torch.zeros(1, d_model)
steered = x0 + 0.8 * true_dirs[3]
with torch.no_grad():
    f_steer = code(steered)
top = f_steer[0].argmax().item()
print(f"adding 0.8 x feature 3 to a zero activation: the TopK SAE's top latent has cos {(dec[top] @ true_dirs[3]).item():.2f} to feature 3")
