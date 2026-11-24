import torch
torch.manual_seed(0)
def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, prev_ids=None, rep_penalty=1.0):
    logits = logits.clone()
    if prev_ids is not None and rep_penalty != 1.0:
        s = logits[prev_ids]; logits[prev_ids] = torch.where(s > 0, s / rep_penalty, s * rep_penalty)
    if temperature == 0: return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(top_k).values[-1]; logits[logits < kth] = float("-inf")
    probs = logits.softmax(-1)
    if min_p > 0:
        probs[probs < min_p * probs.max()] = 0
    if top_p < 1.0:
        sp, si = probs.sort(descending=True); cum = sp.cumsum(0)
        drop = cum - sp > top_p; sp[drop] = 0; probs = torch.zeros_like(probs).scatter(0, si, sp)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))
logits = torch.tensor([3.0, 2.5, 1.0, 0.5, -1.0, -2.0])
for kw in [dict(temperature=0), dict(temperature=1.0), dict(temperature=0.5), dict(top_k=2), dict(top_p=0.8), dict(min_p=0.2)]:
    c = torch.bincount(torch.tensor([sample_next(logits, **kw) for _ in range(5000)]), minlength=6)
    print(kw, (c / 5000).tolist())
print("softmax", [round(x, 3) for x in logits.softmax(-1).tolist()])

def kv_cache_gb(layers, kv_heads, head_dim, seq, batch, bytes_=2):
    return 2 * layers * kv_heads * head_dim * seq * batch * bytes_ / 1e9
print("llama3-8b GQA 8k x1:", round(kv_cache_gb(32, 8, 128, 8192, 1), 2), "GB; x32:", round(kv_cache_gb(32, 8, 128, 8192, 32), 1))
print("if MHA 32 heads:", round(kv_cache_gb(32, 32, 128, 8192, 1), 2))
print("weights 8B fp16:", 8e9*2/1e9, "int4:", 8e9*0.5/1e9)

def quant_int8(w):
    scale = w.abs().max() / 127; q = torch.round(w / scale).clamp(-127, 127).to(torch.int8); return q, scale
w = torch.randn(256, 256); q, s = quant_int8(w); err = (q.float() * s - w).abs().mean() / w.abs().mean()
w2 = w.clone(); w2[0, 0] = 40.0; q2, s2 = quant_int8(w2); err2 = ((q2.float() * s2 - w2).abs()[1:]).mean() / w2.abs()[1:].mean()
def quant_rowwise(w):
    scale = w.abs().amax(1, keepdim=True) / 127; return torch.round(w / scale).clamp(-127, 127), scale
q3, s3 = quant_rowwise(w2); err3 = ((q3 * s3 - w2).abs()[1:]).mean() / w2.abs()[1:].mean()
print("int8 rel err", round(err.item(), 4), "with outlier", round(err2.item(), 4), "row-wise", round(err3.item(), 4))

def spec_expected_tokens(alpha, k): return (1 - alpha ** (k + 1)) / (1 - alpha)
for a in [0.6, 0.8]:
    print("alpha", a, [round(spec_expected_tokens(a, k), 2) for k in [2, 4, 8]])

# decode throughput rough bound: memory bandwidth / bytes per token
bw = 2.0e12; wbytes = 8e9 * 2
print("single stream tok/s bound", round(bw / wbytes, 1))

# ---- additions: speculative decoding, quantisation maths, serving arithmetic ----
import math
import numpy as np

# 21.7 losslessness: the three-token example and a Monte Carlo check
p = np.array([0.6, 0.3, 0.1]); q = np.array([0.3, 0.5, 0.2])
alpha = np.minimum(p, q).sum(); resid = np.maximum(p - q, 0); resid /= resid.sum()
print("spec example: alpha", round(alpha, 3), "residual", resid.round(3).tolist(),
      "P(x) =", (np.minimum(p, q) + (1 - alpha) * resid).round(3).tolist())
rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(10)); Q = rng.dirichlet(np.ones(10))
def spec_step(P, Q, residual=True):
    x = rng.choice(10, p=Q)
    if rng.random() < min(1, P[x] / Q[x]):
        return x
    r = np.maximum(P - Q, 0); r /= r.sum()
    return rng.choice(10, p=r if residual else P)
out = np.bincount([spec_step(P, Q) for _ in range(100_000)], minlength=10) / 100_000
bad = np.bincount([spec_step(P, Q, residual=False) for _ in range(100_000)], minlength=10) / 100_000
print("monte carlo max |freq - p|: correct rule %.4f, wrong rule (resample from p) %.4f; alpha %.3f"
      % (np.abs(out - P).max(), np.abs(bad - P).max(), np.minimum(P, Q).sum()))

# 21.7 wall-clock speedup including the draft cost, and the best k
def speedup(a, k, c): return (1 - a ** (k + 1)) / ((1 - a) * (1 + k * c))
print("speedup alpha 0.8, k 4, c 0.05:", round(speedup(0.8, 4, 0.05), 2))
for a in (0.6, 0.8):
    row = []
    for c in (0.02, 0.05, 0.1, 0.2):
        k = max(range(1, 30), key=lambda k: speedup(a, k, c)); row.append((c, k, round(speedup(a, k, c), 2)))
    print("alpha", a, "best (c, k, speedup):", row)

# 21.2 beam length normalisation
print("beam: unnormalised", 5 * -0.5, 20 * -0.3, "| alpha=1:", -0.5, -0.3)

# 21.3 masking is not conditioning
print("true conditional on branch a: %.2f vs masked 0.60" % (0.6 * 0.1 / (0.6 * 0.1 + 0.4 * 0.9)))

# 21.8 group-wise overhead, GPTQ toy, SmoothQuant
print("bits/weight g=128:", 4 + 16 / 128, "with zero point:", 4 + 32 / 128, "| 8B at 4.125 bits:", 8e9 * 4.125 / 8 / 1e9, "GB")
C = np.array([[1, 0.9], [0.9, 1]]); w = np.array([0.6, 0.6])
loss = lambda e: e @ C @ e
Hinv = np.linalg.inv(C); w2 = w[1] - (w[0] - 1.0) / Hinv[0, 0] * Hinv[1, 0]
print("GPTQ toy: RTN loss %.3f; updated w2 %.2f -> %d; GPTQ loss %.3f"
      % (loss(w - np.round(w)), w2, round(w2), loss(w - np.array([1.0, round(w2)]))))
s_sq = math.sqrt(60) / math.sqrt(0.5)
print("SmoothQuant s = %.2f; new maxima %.2f and %.2f" % (s_sq, 60 / s_sq, 0.5 * s_sq))
print("decode step 8B: fp16 %.1f ms, W4 g128 %.2f ms" % (16e9 / 3.35e12 * 1e3, 8e9 * 4.125 / 8 / 3.35e12 * 1e3))

# 21.9 TTFT from prefill FLOPs (8B: L=32, d=4096) at an achieved 500 TFLOP/s
for T in (2048, 8192, 32768):
    lin, att = 2 * 8e9 * T, 2 * 32 * T * T * 4096
    print(f"prefill {T}: weights {lin:.2e} FLOPs, attention +{att/lin:.0%}, TTFT {lin/500e12:.3f} s -> {(lin+att)/500e12:.2f} s")
print("M/M/1 wait in service times:", {r: round(r / (1 - r), 1) for r in (0.5, 0.8, 0.9, 0.95)})

# 21.6 multi-LoRA adapter size (rank 16, all linear layers of a Llama-3-8B shape), KV offload
r, d, kvd, ff = 16, 4096, 1024, 14336
lora = 32 * r * ((d + d) + 2 * (d + kvd) + (d + d) + 3 * (d + ff))
print(f"LoRA r=16 params {lora/1e6:.1f}M, {lora*2/1e6:.0f} MB; 8k KV reload at 25 GB/s: {1.07/25*1e3:.0f} ms")
print("TP all-reduces at decode, 70B: 160 x 10 us =", 160 * 10e-3, "ms")

# Interview Q5, Q8, Q9 and Practice answer keys
kv = lambda L, h, dh, T, b, by=2: 2 * L * h * dh * T * b * by / 1e9
print("32 users at 32k: fp16 %.0f GB, fp8 %.0f GB; users at 32k in 60 GB: %d"
      % (kv(32, 8, 128, 32768, 32), kv(32, 8, 128, 32768, 32, 1), 60 // kv(32, 8, 128, 32768, 1)))
print("70B 8k %.2f GB, 32k %.2f GB; Qwen2.5-7B 8k %.2f GB" % (kv(80, 8, 128, 8192, 1), kv(80, 8, 128, 32768, 1), kv(28, 4, 128, 8192, 1)))
print("p95 breakdown:", 0.5 + 0.4 + 1.1 + 350 * 0.02, "s; static batch utilisation", (7 * 100 + 500) / (8 * 500))
tokens = 50e6 * 600; tps = 400e12 / (2 * 8e9); hours = tokens / tps / 3600
var_per_m = 2.5 / (tps * 3600 * 0.4) * 1e6
print(f"Q9: {tokens:.0e} tokens; API $1.5/M ${tokens/1e6*1.5:,.0f}, $0.075/M ${tokens/1e6*0.075:,.0f}; "
      f"self-host {hours:.0f} GPU-h, {hours/0.4:.0f} at 40% = ${hours/0.4*2.5:,.0f}; "
      f"variable ${var_per_m:.3f}/M; break-even {15000/(1.5-var_per_m)/1e3:.1f}B tokens/month")
print("30 problems at 60%%, one sample: SE %.1f points" % (100 * math.sqrt(0.24 / 30)))
