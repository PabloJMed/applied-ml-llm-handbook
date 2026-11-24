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

# 18.4 policy-gradient foundations: REINFORCE with and without baselines on a one-prompt bandit
# Four possible responses; the policy is a softmax over logits theta. Exact gradient of E[r]:
# d/dtheta E[r] = pi * (r - E[r]). REINFORCE estimates it from N samples as mean (R_i - b) * grad log pi(a_i),
# with grad log pi(a) = onehot(a) - pi.
pg_rng = np.random.default_rng(18)      # own generator: leaves the later sections' numbers unchanged
theta = np.array([0.5, 0.2, -0.1, -0.6]); pi = np.exp(theta) / np.exp(theta).sum()
r_true = np.array([1.0, 0.6, 0.2, 0.0])
exact = pi * (r_true - pi @ r_true)
def reinforce(r, N, baseline, trials=20000):
    a = pg_rng.choice(4, size=(trials, N), p=pi)
    R = r[a]                                                     # (trials, N)
    if baseline == "none":
        b = 0.0
    elif baseline == "group mean":                               # GRPO-style, includes the sample itself
        b = R.mean(1, keepdims=True)
    else:                                                        # leave-one-out (RLOO)
        b = (R.sum(1, keepdims=True) - R) / (N - 1)
    glp = np.eye(4)[a] - pi                                      # (trials, N, 4)
    g = ((R - b)[..., None] * glp).mean(1)                      # (trials, 4)
    return g
print("exact policy gradient:", np.round(exact, 4).tolist())
for shift in (0.0, 10.0):
    for base in ("none", "group mean", "leave-one-out"):
        g = reinforce(r_true + shift, 8, base)
        bias = np.abs(g.mean(0) - exact).max()
        sd = np.sqrt(g.var(0).sum())
        print(f"rewards +{shift:4.1f}, baseline {base:13s}: max |bias| {bias:.4f}  total std {sd:.3f}  "
              f"(mean/exact ratio {np.dot(g.mean(0), exact) / np.dot(exact, exact):.3f})")

# 18.4 importance sampling over a sequence: per-token ratios multiply, so the variance explodes with length.
# Per-token log-ratio ~ N(-s^2/2, s^2) (mean ratio 1). Sequence ratio variance = exp(T s^2) - 1;
# effective sample size per sample = 1 / (1 + variance).
s = 0.03
for T in (100, 1000, 4000):
    var = math.exp(T * s * s) - 1
    w = np.exp(pg_rng.normal(-T * s * s / 2, math.sqrt(T) * s, 200_000))
    ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
    print(f"T={T:5d} tokens, per-token log-ratio std {s}: sequence-ratio variance {var:.2f}, "
          f"effective sample size {1 / (1 + var):.3f} (simulated {ess:.3f})")

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

# ---------------------------------------------------------------------------------------------
# Additions: derivations and worked numbers added in the review pass
# ---------------------------------------------------------------------------------------------
print("\n--- additions ---")

# 18.3 ties: a soft 0.5 label pulls the reward difference to zero
for dd in (0.0, 0.5, 1.0):
    tie = -(0.5 * F.logsigmoid(torch.tensor(dd)) + 0.5 * F.logsigmoid(torch.tensor(-dd)))
    print(f"tie loss at reward difference {dd}: {tie.item():.3f}")
print("annotation budget: 100k pairs x 4 min = %.0f annotator-hours" % (100_000 * 4 / 60))

# 18.4 GAE as a bias-variance dial: same trajectory at lambda = 0, 0.95, 1
def gae(lam):
    adv, last = [0.0] * 4, 0.0
    for t in reversed(range(4)):
        delta = rewards[t].item() + values[t + 1].item() - values[t].item()
        last = delta + lam * last; adv[t] = last
    return [round(a, 3) for a in adv]
for lam_ in (0.0, 0.95, 1.0):
    print(f"GAE lambda={lam_}: {gae(lam_)}")
# lambda as a bias-variance dial, by simulation. A 20-token episode; the true value of a state is p_t, the
# probability the answer ends up correct, a martingale (p moves up or down by c*min(p, 1-p) per token).
# The reward is Bernoulli(p_T) at the end. The first token is either "good" (p rises) or "bad" (p falls),
# so the true advantage gap between them is 0.30. The value head is systematically under-confident:
# V_hat(p) = 0.5 + 0.6 (p - 0.5). The policy-gradient signal for the first token is E[A | good] - E[A | bad].
sim = np.random.default_rng(7)
def gae_first(lam, first_up, n_ep=40000, T_=20, c=0.3):
    p = np.full(n_ep, 0.5); ps = [p.copy()]
    for t in range(T_):
        up = np.full(n_ep, first_up) if t == 0 else sim.random(n_ep) < 0.5
        p = p + np.where(up, 1, -1) * c * np.minimum(p, 1 - p); ps.append(p.copy())
    R = (sim.random(n_ep) < p).astype(float)
    V = [0.5 + 0.6 * (q - 0.5) for q in ps[:T_]] + [np.zeros(n_ep)]   # V(s_0..s_{T-1}), terminal 0
    adv = np.zeros(n_ep)
    for t in reversed(range(T_)):
        r_t = R if t == T_ - 1 else 0.0
        adv = r_t + V[t + 1] - V[t] + lam * adv
    return adv
for lam_ in (0.0, 0.5, 0.9, 0.95, 1.0):
    a_up, a_dn = gae_first(lam_, True), gae_first(lam_, False)
    print(f"  lambda={lam_:4}: signal E[A|good]-E[A|bad] = {a_up.mean() - a_dn.mean():.3f} (true 0.300), "
          f"std of A within an action {np.sqrt((a_up.var() + a_dn.var()) / 2):.3f}")

# 18.5 best-of-n KL: log n - (n-1)/n is exact for continuous rewards (no ties), an upper bound otherwise
for n in (4, 16):
    u = rng.random((200_000, n)).max(1)                     # CDF value of the max ~ Beta(n, 1)
    print(f"best-of-{n}: E[log(n u^(n-1))] = {np.mean(np.log(n) + (n - 1) * np.log(u)):.3f}, formula {math.log(n) - (n - 1) / n:.3f}")

# 18.6 the DPO identity: J(pi) = -beta KL(pi || pi*) + beta log Z, checked on the bandit
r4 = torch.tensor([1.0, 0.5, 0.0, -1.0]); beta4 = 0.5; ref4 = torch.tensor([0.1, 0.2, 0.3, 0.4])
Z = (ref4 * torch.exp(r4 / beta4)).sum(); pstar = ref4 * torch.exp(r4 / beta4) / Z
def J(pi):  return (pi * r4).sum() - beta4 * (pi * (pi / ref4).log()).sum()
def kl(p, q): return (p * (p / q).log()).sum()
for name, pi_ in (("pi_ref", ref4), ("uniform", torch.full((4,), 0.25)), ("pi*", pstar)):
    print(f"J({name}) = {J(pi_).item():+.4f};  -beta KL(pi||pi*) + beta log Z = {(-beta4 * kl(pi_, pstar) + beta4 * Z.log()).item():+.4f}")
print(f"Z = {Z.item():.4f}, beta log Z = {(beta4 * Z.log()).item():.4f}, pi* = {[round(v, 3) for v in pstar.tolist()]}")
print("implied reward from pi*: beta log(pi*/pi_ref) + beta log Z =",
      [round(v, 3) for v in (beta4 * (pstar / ref4).log() + beta4 * Z.log()).tolist()])

# 18.6 gradient weight of one DPO pair and the family of losses on the same pair (h = 1.5)
b = 0.1
print(f"DPO pair: h = {h.item():.2f}, loss {-F.logsigmoid(b * h).item():.3f}, gradient weight sigma(-beta h) = {torch.sigmoid(-b * h).item():.3f}")
eps = 0.1
cdpo = (1 - eps) * -F.logsigmoid(b * h) + eps * -F.logsigmoid(-b * h)
print(f"cDPO (eps 0.1): {cdpo.item():.3f}; IPO (tau 0.1): target margin {1/(2*b):.1f}, loss {(h - 1/(2*b))**2:.2f}")
print(f"SimPO per-token log-probs: chosen {pc.item()/len_c:.3f}, rejected {pr.item()/len_r:.3f}; "
      f"loss {simpo.item():.3f}")
# KTO: desirable / undesirable, z0 = reference point (batch KL estimate, no gradient)
z0, lam_d, lam_u = 0.2, 1.0, 1.0
rc_ratio, rr_ratio = (pc - rc).item(), (pr - rr).item()
kto_d = lam_d * (1 - torch.sigmoid(torch.tensor(b * (rc_ratio - z0))))
kto_u = lam_u * (1 - torch.sigmoid(torch.tensor(b * (z0 - rr_ratio))))
print(f"KTO (beta 0.1, z0 0.2): desirable log-ratio {rc_ratio:+.1f} -> {kto_d.item():.3f}; "
      f"undesirable log-ratio {rr_ratio:+.1f} -> {kto_u.item():.3f}")
# ORPO: odds ratio on length-normalised likelihoods, added to the SFT loss
Pc, Pr = math.exp(pc.item() / len_c), math.exp(pr.item() / len_r)
lor = math.log(Pc / (1 - Pc)) - math.log(Pr / (1 - Pr))
print(f"ORPO: P_c {Pc:.3f}, P_r {Pr:.3f}, log odds ratio {lor:+.3f}, odds-ratio loss {-math.log(1/(1+math.exp(-lor))):.3f}")
