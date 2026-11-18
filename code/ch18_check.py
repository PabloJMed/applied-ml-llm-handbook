"""Chapter 18 · Post-training I: SFT, reward models and preference optimisation (CPU, seconds)."""
import math
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
rng = np.random.default_rng(0)

# 18.2 packing: two examples in one row, with position ids reset and a block-diagonal causal mask
lens = [5, 3]
pos = torch.cat([torch.arange(n) for n in lens])
doc = torch.cat([torch.full((n,), i) for i, n in enumerate(lens)])
mask = (doc[:, None] == doc[None, :]) & (torch.arange(8)[:, None] >= torch.arange(8)[None, :])
print("position ids:", pos.tolist(), " row 6 attends to:", mask[6].nonzero().flatten().tolist())

# 18.3 a Bradley-Terry reward model on synthetic preferences
d = 8
w_true = torch.randn(d)
X_c, X_r = torch.randn(2000, d), torch.randn(2000, d)
p_pref = torch.sigmoid(X_c @ w_true - X_r @ w_true)
swap = torch.rand(2000) > p_pref                       # labellers prefer the higher-reward response, noisily
chosen = torch.where(swap[:, None], X_r, X_c); rejected = torch.where(swap[:, None], X_c, X_r)
w = torch.zeros(d, requires_grad=True)
opt = torch.optim.Adam([w], lr=0.05)
for _ in range(300):
    loss = -F.logsigmoid(chosen[:1500] @ w - rejected[:1500] @ w).mean()
    opt.zero_grad(); loss.backward(); opt.step()
acc = ((chosen[1500:] @ w) > (rejected[1500:] @ w)).float().mean()
ceiling = torch.maximum(p_pref, 1 - p_pref)[1500:].mean()
print(f"reward model held-out pairwise accuracy {acc:.3f} (label-noise ceiling {ceiling:.3f}); "
      f"cosine to true reward {F.cosine_similarity(w, w_true, 0):.3f}")

# 18.4 GAE and the PPO clipped objective on one trajectory
rewards = torch.tensor([0., 0., 0., 1.0])              # reward only at the end (the reward model's score)
values = torch.tensor([0.4, 0.5, 0.6, 0.8, 0.0])       # V(s_t) from the value head; terminal value 0
gamma, lam = 1.0, 0.95
adv, last = torch.zeros(4), 0.0
for t in reversed(range(4)):
    delta = rewards[t] + gamma * values[t + 1] - values[t]
    last = delta + gamma * lam * last
    adv[t] = last
print("GAE advantages:", [round(v, 3) for v in adv.tolist()], " returns:", [round(v, 3) for v in (adv + values[:4]).tolist()])
def ppo_clip(ratio, A, eps=0.2):
    return torch.minimum(ratio * A, ratio.clamp(1 - eps, 1 + eps) * A)
for r, A in ((1.5, 1.0), (0.5, 1.0), (1.5, -1.0), (0.5, -1.0)):
    print(f"ratio {r}, advantage {A:+.0f}: objective {ppo_clip(torch.tensor(r), torch.tensor(A)).item():+.2f}")

# 18.4 three KL estimators from samples of the policy (Schulman's k1, k2, k3)
p = torch.tensor([0.5, 0.3, 0.2]); q = torch.tensor([0.4, 0.4, 0.2])   # policy p, reference q
true_kl = (p * (p / q).log()).sum().item()
s = torch.multinomial(p, 100_000, replacement=True)
logr = (q[s] / p[s]).log()
for name, est in (("k1 = -log r", -logr), ("k2 = (log r)^2 / 2", logr ** 2 / 2), ("k3 = r - 1 - log r", logr.exp() - 1 - logr)):
    print(f"{name:20s} mean {est.mean():.4f} std {est.std():.4f}  (true KL {true_kl:.4f})")

# 18.5 over-optimisation: best-of-n against a proxy reward that partly rewards length
n_samples = 200_000
quality = rng.normal(size=n_samples); length = rng.normal(size=n_samples)
gold = quality - 0.5 * length ** 2                 # people dislike answers far from the right length
proxy = quality + 1.0 * length                     # the reward model also likes long answers
for n in (1, 4, 16, 64, 256, 1024):
    k = n_samples // n
    idx = proxy[: k * n].reshape(k, n).argmax(1) + np.arange(k) * n
    kl = math.log(n) - (n - 1) / n
    print(f"best-of-{n:<4}: KL {kl:5.2f} nats  proxy {proxy[idx].mean():5.2f}  gold {gold[idx].mean():5.2f}")

# 18.6 DPO on a bandit recovers the closed-form optimum pi* ~ pi_ref * exp(r / beta)
r = torch.tensor([1.0, 0.5, 0.0, -1.0]); beta = 0.5
pi_ref = torch.tensor([0.1, 0.2, 0.3, 0.4])
target = pi_ref * torch.exp(r / beta); target /= target.sum()
logits = pi_ref.log().clone().requires_grad_(True)
opt = torch.optim.Adam([logits], lr=0.05)
pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
for step in range(3000):
    logp = F.log_softmax(logits, 0)
    loss = 0.0
    for i, j in pairs:                               # preference probability from Bradley-Terry on the true r
        p_ij = torch.sigmoid(r[i] - r[j])
        margin = beta * ((logp[i] - pi_ref[i].log()) - (logp[j] - pi_ref[j].log()))
        loss = loss - p_ij * F.logsigmoid(margin)
    opt.zero_grad(); (loss / len(pairs)).backward(); opt.step()
print("closed-form optimum:", [round(v, 3) for v in target.tolist()])
print("DPO solution:       ", [round(v, 3) for v in F.softmax(logits, 0).tolist()])

# 18.6 DPO, IPO and SimPO losses on the same numbers
pc, pr, rc, rr = torch.tensor(-20.0), torch.tensor(-22.0), torch.tensor(-21.0), torch.tensor(-21.5)
h = (pc - rc) - (pr - rr)
print(f"log-ratio margin h = {h.item():.2f}: DPO {-F.logsigmoid(0.1 * h).item():.3f}  "
      f"IPO {(h - 1 / (2 * 0.1)).pow(2).item():.3f}")
len_c, len_r = 40, 60
simpo = -F.logsigmoid(2.0 * (pc / len_c - pr / len_r) - 0.5)
print(f"SimPO (beta 2, margin 0.5, length-normalised, no reference): {simpo.item():.3f}")
