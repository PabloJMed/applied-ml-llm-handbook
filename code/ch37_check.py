"""Checks for the worked numbers in Chapter 37 (ML depth question bank).

Run: python3 code/ch37_check.py
Each block prints the number quoted in the chapter, with the question it belongs to.
"""
import math
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)


def hr(t):
    print(f"\n== {t}")


# ---------------------------------------------------------------- Q60 / Chain 1: sqrt(d_k)
hr("Q60  q.k variance grows with d_k")
for dk in (64, 128):
    q = rng.standard_normal((200_000, dk)); k = rng.standard_normal((200_000, dk))
    s = (q * k).sum(1)
    print(f"d_k={dk}: std(q.k)={s.std():.2f} (sqrt d_k={math.sqrt(dk):.2f}); scaled std={(s / math.sqrt(dk)).std():.2f}")
# softmax over 8 keys with logit std 11 vs 1: share of mass on the top key
for sd in (1.0, math.sqrt(128)):
    z = rng.standard_normal((20_000, 8)) * sd
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    print(f"logit std {sd:.1f}: mean max softmax prob over 8 keys = {p.max(1).mean():.2f}")

# ---------------------------------------------------------------- Q63: KV cache
hr("Q63  KV cache = 2 x layers x kv_heads x head_dim x tokens x bytes x batch")
def kv_bytes(layers, kv_heads, head_dim, tokens, nbytes=2, batch=1):
    return 2 * layers * kv_heads * head_dim * tokens * nbytes * batch
per_tok_8b = kv_bytes(32, 8, 128, 1)
print(f"Llama-3-8B-like (32 L, 8 KV heads, d_h 128, bf16): {per_tok_8b/1024:.0f} KiB/token; "
      f"128k tokens: {kv_bytes(32, 8, 128, 131072)/2**30:.0f} GiB")
print(f"same model with 32 KV heads (no GQA): {kv_bytes(32, 32, 128, 131072)/2**30:.0f} GiB")

# ---------------------------------------------------------------- Q78: speculative decoding
hr("Q78  speculative sampling is exact; acceptance = sum min(p, q)")
p = np.array([0.5, 0.3, 0.15, 0.05]); q = np.array([0.25, 0.25, 0.25, 0.25])
alpha = np.minimum(p, q).sum()
res = np.maximum(p - q, 0); res /= res.sum()
out = q * np.minimum(1, p / q) + (1 - alpha) * res
print("target p:", p, " draft q:", q)
print(f"acceptance prob alpha = {alpha:.2f}; output distribution = {out.round(4)} (equals p: {np.allclose(out, p)})")
# expected tokens per target pass with k drafted tokens, i.i.d. acceptance alpha
for a, k in ((0.8, 4), (0.6, 4)):
    e = (1 - a ** (k + 1)) / (1 - a)
    print(f"alpha={a}, k={k}: expected tokens per target forward = {e:.2f}")
# wall-clock with draft cost c (draft forward / target forward)
a, k, c = 0.8, 4, 0.05
e = (1 - a ** (k + 1)) / (1 - a)
print(f"speed-up with draft cost c={c}: {e / (1 + k * c):.2f}x")

# ---------------------------------------------------------------- Q80: arithmetic intensity
hr("Q80 / Q252  roofline")
peak, bw = 989e12, 3.35e12   # H100 SXM dense bf16, HBM3
print(f"H100 ridge point = {peak / bw:.0f} FLOP/byte")
print("batch-1 bf16 matvec: 2 FLOPs per weight / 2 bytes per weight = 1 FLOP/byte")
for B in (1, 64, 256, 512):
    print(f"  batch {B}: intensity ~ {B} FLOP/byte -> {'memory' if B < peak / bw else 'compute'}-bound")
t = 16e9 / bw
print(f"8B bf16 weights (16 GB) read per token at 3.35 TB/s: {t*1e3:.1f} ms -> <= {1/t:.0f} tokens/s at batch 1")

# ---------------------------------------------------------------- Q169 / Chain 8: Chinchilla exponents
hr("Q169  compute-optimal exponents")
E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
print(f"a = beta/(alpha+beta) = {be/(al+be):.3f}; b = alpha/(alpha+beta) = {al/(al+be):.3f}")
def opt(C):
    G = (al * A / (be * B)) ** (1 / (al + be))
    N = G * (C / 6) ** (be / (al + be)); D = C / (6 * N)
    return N, D
for C in (1e21, 1e23, 1e24):
    N, D = opt(C)
    print(f"C={C:.0e}: N_opt={N/1e9:.1f}B, D_opt={D/1e9:.0f}B tokens, D/N={D/N:.0f}")
# numerical check of the analytic optimum
C = 1e23
Ns = np.logspace(9, 12, 20001)
L = E + A / Ns ** al + B / (C / (6 * Ns)) ** be
print(f"grid argmin at C=1e23: {Ns[L.argmin()]/1e9:.1f}B")
# extrapolation error: a 10% error in the exponent gap over 3 decades
print(f"C^0.45 vs C^0.50 over 1000x compute: ratio {1000**0.05:.2f}")

# ---------------------------------------------------------------- Q190 / Chain 7: Switch aux loss
hr("Q190  aux loss alpha * E * sum f_e P_e")
Eexp, alpha_aux = 8, 0.01
f = np.full(Eexp, 1 / Eexp); P = np.full(Eexp, 1 / Eexp)
print(f"uniform: E*sum f P = {Eexp * (f * P).sum():.2f}")
f2 = np.array([0.5] + [0.5 / 7] * 7); P2 = np.array([0.4] + [0.6 / 7] * 7)
print(f"one expert takes half the tokens: E*sum f P = {Eexp * (f2 * P2).sum():.2f}")
# capacity factor: tokens dropped
tokens, cf = 4096, 1.25
cap = int(cf * tokens / Eexp)
load = (f2 * tokens).astype(int)
print(f"capacity per expert = {cap}; dropped tokens with the skewed load = {np.maximum(load - cap, 0).sum()} of {tokens}")

# ---------------------------------------------------------------- Q205: KL estimators
hr("Q205  k1, k2, k3 with r = pi_ref/pi_theta, y ~ pi_theta")
pt = np.array([0.6, 0.3, 0.1]); pr = np.array([0.5, 0.3, 0.2])
true_kl = (pt * np.log(pt / pr)).sum()
y = rng.choice(3, size=200_000, p=pt); r = pr[y] / pt[y]
k1, k2, k3 = -np.log(r), 0.5 * np.log(r) ** 2, r - 1 - np.log(r)
print(f"true KL = {true_kl:.4f}")
for name, k in (("k1", k1), ("k2", k2), ("k3", k3)):
    print(f"{name}: mean {k.mean():.4f}, std {k.std():.4f}, min {k.min():.3f}")

# ---------------------------------------------------------------- Q211: best-of-n KL
hr("Q211  best-of-n KL: log n - (n-1)/n is an upper bound")
for n in (4, 16, 64):
    print(f"n={n}: log n - (n-1)/n = {math.log(n) - (n - 1) / n:.3f}")
# exact KL for a discrete base with ties broken uniformly within a reward level
def bon_kl(probs, rewards, n):
    order = np.argsort(rewards); probs = probs[order]; rewards = rewards[order]
    kl = 0.0
    for lvl in np.unique(rewards):
        m = rewards == lvl
        below = probs[rewards < lvl].sum(); at = probs[m].sum()
        mass = (below + at) ** n - below ** n          # P(best has this reward level)
        for pi in probs[m]:
            pb = mass * pi / at
            if pb > 0: kl += pb * math.log(pb / pi)
    return kl
pb = np.array([0.25, 0.25, 0.25, 0.25]); rw = np.array([0, 1, 2, 3])
print(f"4 equally likely answers, distinct rewards, n=4: exact {bon_kl(pb, rw, 4):.3f} vs bound {math.log(4)-0.75:.3f}")
pb = np.full(1000, 1e-3); rw = np.arange(1000)
print(f"1000 answers, n=4: exact {bon_kl(pb, rw, 4):.3f}")
pb = np.array([0.7, 0.3]); rw = np.array([0, 1])
print(f"binary verifier, 30% correct, n=4: exact {bon_kl(pb, rw, 4):.3f} vs bound {math.log(4)-0.75:.3f}")

# ---------------------------------------------------------------- Q223/Q225, Chain 3: paired sample size
hr("Q225  n = (z_{a/2} + z_b)^2 * d / delta^2")
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
print(f"(1.96+0.84)^2 = {z**2:.2f}")
for d, delta in ((0.10, 0.02), (0.20, 0.01), (0.20, 0.02)):
    print(f"d={d:.2f}, delta={delta:.2f}: n = {z**2 * d / delta**2:,.0f}")
print(f"Chain 3 unpaired at 72%: {2 * z**2 * 0.72 * 0.28 / 0.01**2:,.0f} per model")
print(f"SE one model 500 q at 70%: {math.sqrt(0.7*0.3/500)*100:.2f} points")

# ---------------------------------------------------------------- Q72 / Q251: test-time compute
hr("Q72 / Q251  pass@k and majority vote")
for p1 in (0.2, 0.5):
    for k in (1, 8, 64):
        print(f"p={p1}, k={k}: pass@k = {1 - (1 - p1) ** k:.3f}")
# majority vote with 3 answer options: correct w.p. 0.4, two wrong answers 0.35 and 0.25
def maj(pc, pw, k, trials=40_000):
    probs = np.array([pc] + pw)
    draws = rng.multinomial(k, probs, size=trials)
    top = draws.max(1, keepdims=True); win = (draws == top)
    return (win[:, 0] / win.sum(1)).mean()
print(f"majority@16, correct 0.40 vs wrong 0.35/0.25: {maj(0.40, [0.35, 0.25], 16):.2f}")
print(f"majority@16, correct 0.30 vs wrong 0.45/0.25: {maj(0.30, [0.45, 0.25], 16):.2f}")

# ---------------------------------------------------------------- Q259: FLOPs per token with attention
hr("Q259  training FLOPs/token = 6N + 12 L d T (PaLM count, no causal saving)")
L_, d_, N_ = 32, 4096, 8e9
for T in (8192, 32768, 131072):
    att = 12 * L_ * d_ * T
    print(f"8B-like, T={T}: 6N={6*N_:.2e}, attention={att:.2e}, ratio={att/(6*N_):.2f}")
print(f"crossover T = 6d for N = 12 L d^2: d=4096 -> T = {6*4096}")

# ---------------------------------------------------------------- H: 70B GQA KV cache and decode
hr("H  70B GQA serving estimate")
per_tok = kv_bytes(80, 8, 128, 1)
print(f"Llama-3-70B-like (80 L, 8 KV heads, d_h 128): {per_tok/1024:.0f} KiB/token; "
      f"32k tokens: {kv_bytes(80, 8, 128, 32768)/2**30:.1f} GiB = {kv_bytes(80, 8, 128, 32768)/1e9:.1f} GB")
weights = 70e9 * 2
gpus = 4; hbm = 80e9 * gpus
free = hbm - weights
print(f"4x80 GB: weights {weights/1e9:.0f} GB, left {free/1e9:.0f} GB -> "
      f"{free / kv_bytes(80, 8, 128, 32768):.1f} sequences at 32k (before activations/fragmentation)")
batch = 8
read = weights + batch * kv_bytes(80, 8, 128, 32768)
step = read / (gpus * bw)
print(f"batch {batch} at 32k: bytes per step {read/1e9:.0f} GB, step {step*1e3:.1f} ms, "
      f"{1/step:.0f} tok/s per sequence, {batch/step:.0f} tok/s total")
flops = 2 * 70e9 * batch + batch * 4 * 80 * 8192 * 32768  # weights + attention (64 q heads x 128 = 8192)
print(f"compute per step {flops:.2e} FLOPs -> {flops/(gpus*peak)*1e3:.2f} ms at peak: memory-bound")
print(f"without GQA (64 KV heads) KV per sequence at 32k: {kv_bytes(80, 64, 128, 32768)/1e9:.0f} GB")

# ---------------------------------------------------------------- Q243 (existing)
hr("Q243  variance of a product of T ratios")
print(f"exp(T sigma^2)-1 at sigma=0.03, T=4000: {math.exp(4000*0.03**2)-1:.1f}; ESS ~ {1/math.exp(4000*0.03**2):.3f}")

# ---------------------------------------------------------------- Q239 (existing)
hr("Q239  rule of three")
print(f"0/100 -> 95% upper bound {1 - 0.05 ** (1/100):.3f} (3/n = 0.03)")

# ---------------------------------------------------------------- H: prefill time for the 70B example
hr("H  70B prefill of a 32k prompt on 4 x H100")
T, L, d = 32768, 80, 8192
lin = 2 * 70e9 * T
att = 2 * L * T ** 2 * d           # causal: QK^T and AV, 4 t d per token at position t, summed over t
tot = lin + att
print(f"linear {lin:.2e} + attention {att:.2e} = {tot:.2e} FLOPs; "
      f"{tot / (4 * peak):.2f} s at peak, {tot / (4 * peak * 0.5):.1f} s at 50% MFU")

# ---------------------------------------------------------------- H: DPO loss and gradient weight
hr("H  DPO: loss and gradient weight at beta = 0.1")
sig = lambda x: 1 / (1 + math.exp(-x))
for gap in (0, 5, 20):
    print(f"log-ratio gap {gap:>2} nats: loss {-math.log(sig(0.1 * gap)):.3f}, gradient weight sigma(-beta*gap) {sig(-0.1 * gap):.3f}")

# ---------------------------------------------------------------- Chain 8: tokens per parameter for an overtrained small model
hr("Chain 8  15T tokens on an 8B model")
print(f"{15e12 / 8e9:,.0f} tokens per parameter")
