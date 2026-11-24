"""Appendix E (worked system designs): every number quoted in E.1-E.15 that is computed, not assumed.

    python code/appE_check.py

Prices are the appendix's placeholders: $3 / $15 per million input / output tokens, $0.30 per
million cached input tokens, $2.50 per H100-hour. Hardware: H100 SXM, 80 GB, 3.35 TB/s, 989 TFLOP/s
dense bf16.
"""
import math

IN, OUT, CACHED, GPU_HOUR = 3.0, 15.0, 0.30, 2.50
BW, PEAK = 3.35e12, 989e12


def se(p, n):
    return math.sqrt(p * (1 - p) / n)


print("== E.1 feed / B.5 QPS")
print(f"10M users x 5 requests: {10e6 * 5 / 86400:.0f} QPS average, {3 * 10e6 * 5 / 86400:.0f} at peak")

print("== E.3 taxonomy evaluation set")
print(f"items per leaf: {2500 / 800:.2f}")
print(f"leaf accuracy SE at 85%, n=2,500: {100 * se(0.85, 2500):.2f} points (95% +-{196 * se(0.85, 2500):.2f})")
print(f"one leaf with 30 items at 80%: 95% +-{196 * se(0.8, 30):.1f} points")

print("== E.4 RAG assistant")
per_conv = 3 * (3000 * IN + 200 * OUT) / 1e6
print(f"cost per conversation ${per_conv:.3f}; per day at 100k: ${per_conv * 100e3:,.0f}")
print(f"share of input saved by caching a 1,000-token prefix at 10% price: {1000 * 0.9 / 3000:.0%}")
print(f"500 items, 95% CI half-width: p=0.5 {196 * se(0.5, 500):.1f}, p=0.8 {196 * se(0.8, 500):.1f} points")

print("== E.5 moderation queue: 2% of 300k =", int(0.02 * 300e3))

print("== E.7 GSP auction and normalised entropy")
ads = {"A": (2.00, 0.03), "B": (3.00, 0.015), "C": (1.00, 0.03)}
score = {k: b * c for k, (b, c) in ads.items()}
order = sorted(score, key=score.get, reverse=True)
for slot in range(2):
    k, nxt = order[slot], order[slot + 1]
    print(f"slot {slot + 1}: {k} score {score[k]:.3f}, pays {score[nxt] / ads[k][1]:.2f} per click (bid {ads[k][0]:.2f})")
pbar = 0.02
h = -(pbar * math.log(pbar) + (1 - pbar) * math.log(1 - pbar))
print(f"baseline entropy at CTR 2%: {h:.3f} nats; NE for log loss 0.085: {0.085 / h:.2f}")

print("== E.8 newsvendor quantile with 4:1 costs:", 4 / (4 + 1))

print("== E.10 agent cost and latency")
case = 6 * (5000 * IN + 300 * OUT) / 1e6
print(f"cost per case ${case:.3f}; per day at 40k ${case * 40e3:,.0f}; latency {6 * (1.5 + 0.3):.1f} s")
print(f"pass^3 at 0.9 per run: {0.9 ** 3:.3f}")

print("== E.11 serving a 70B dense GQA model")
L, KVH, HD, N = 80, 8, 128, 70e9
kv_tok = 2 * L * KVH * HD * 2
avg_qps = 20e6 / 86400
peak_qps = 3 * avg_qps
out_tps = peak_qps * 400
prefill_tps = peak_qps * 1500 * 0.5
print(f"KV per token {kv_tok / 1e3:.0f} KB; per 1,900-token request {1900 * kv_tok / 1e6:.0f} MB")
print(f"QPS avg {avg_qps:.0f}, peak {peak_qps:.0f}; peak output {out_tps / 1e3:.0f}k tok/s, prefill {prefill_tps / 1e3:.0f}k tok/s")
kv_free = 8 * 80e9 * 0.9 - N * 2
print(f"KV memory free {kv_free / 1e9:.0f} GB -> {kv_free / (1900 * kv_tok):.0f} sequences")


def decode(batch, wbytes, live=1700):
    kv_read = batch * live * kv_tok * (wbytes / 2)
    t_mem = (N * wbytes + kv_read) / (8 * BW * 0.7)
    t_cmp = 2 * N * batch / (8 * PEAK * 0.5 * (2 / wbytes))
    return kv_read, t_mem, t_cmp, batch / max(t_mem, t_cmp)


kv_read, t_mem, t_cmp, tps = decode(256, 2)
print(f"bf16 b=256: KV read {kv_read / 1e9:.0f} GB, step {t_mem * 1e3:.1f} ms (compute {t_cmp * 1e3:.1f} ms, "
      f"ratio {t_cmp / t_mem:.2f}), {tps:,.0f} tok/s/replica")
dec_rep = out_tps / tps
pre_rep = prefill_tps * 2 * N / (8 * PEAK * 0.5)
print(f"decode replicas {dec_rep:.1f}, prefill replicas {pre_rep:.1f}, total GPUs {8 * (math.ceil(dec_rep) + math.ceil(pre_rep))}")
print(f"Little's law in flight: {peak_qps * (0.3 + 400 * 0.015):,.0f} sequences -> {peak_qps * (0.3 + 400 * 0.015) / 256:.1f} replicas")
_, _, _, tps8 = decode(256, 1)
dec8, pre8 = out_tps / tps8, pre_rep / 2
gpus8 = 8 * (math.ceil(dec8) + math.ceil(pre8))
print(f"fp8: {tps8:,.0f} tok/s/replica, decode replicas {dec8:.1f}, prefill {pre8:.1f}, GPUs {gpus8}")
day = 288 * 24 * GPU_HOUR
print(f"cost at peak size ${day:,.0f}/day; per M output tokens ${day / (20e6 * 400 / 1e6):.2f}; "
      f"perfect autoscaling ${day / 3 / (20e6 * 400 / 1e6):.2f}; fp8 saving ${(288 - gpus8) * 24 * GPU_HOUR:,.0f}/day")

print("== E.12 RLVR pipeline")
for p in (0.5, 0.2, 0.8, 0.9, 0.95):
    print(f"G=16, pass rate {p}: no-signal probability {p ** 16 + (1 - p) ** 16:.3f}")
gen = 512 * 16 * 8000
tokens = 512 * 16 * 8500
t_gen = gen / (128 * 1000)
t_train = 10 * 32e9 * tokens / (128 * PEAK * 0.4)
print(f"generated tokens/step {gen / 1e6:.1f}M; generation {t_gen:.0f} s; training FLOPs {10 * 32e9 * tokens:.2e}, {t_train:.0f} s")
print(f"sync: {(t_gen + t_train):.0f} s/step, 1,000 steps {1000 * (t_gen + t_train) / 86400:.1f} days; "
      f"async: {max(t_gen, t_train):.0f} s/step, {1000 * max(t_gen, t_train) / 86400:.1f} days")
print(f"max SE: 500 problems {100 * se(0.5, 500):.1f} points, 30 problems {100 * se(0.5, 30):.1f} points")

print("== E.13 pretraining data pipeline")
print(f"text {15e12 * 4.2 / 1e12:.0f} TB, token ids {15e12 * 4 / 1e12:.0f} TB, MinHash signatures {5e9 * 128 * 4 / 1e12:.2f} TB")
b, r = 16, 8
print(f"LSH threshold (1/b)^(1/r) = {(1 / b) ** (1 / r):.3f}")
for s in (0.5, 0.7, 0.8, 0.9):
    print(f"  Jaccard {s}: candidate probability {1 - (1 - s ** r) ** b:.4f}")
R_STAR = 15.4
for epochs in (4, 16):
    eff = 1 + R_STAR * (1 - math.exp(-(epochs - 1) / R_STAR))
    print(f"  {epochs} epochs worth {eff / epochs:.0%} of fresh data")
abl = 6 * 1e9 * 25e9
print(f"ablation 1B x 25B: {abl:.2e} FLOPs, {abl / (64 * PEAK * 0.4) / 3600:.1f} h on 64 H100s at 40% MFU")
print(f"SE at 40%, n=1,000: {100 * se(0.4, 1000):.2f} points")

print("== E.14 evaluation platform")
print(f"paired n for 2 points at 10% disagreement: {(1.96 + 0.84) ** 2 * 0.10 / 0.02 ** 2:.0f}")
print(f"40 tests at 0.05: expected false flags {40 * 0.05:.0f}, P(at least one) {1 - 0.95 ** 40:.2f}")
for k in (1, 4, 8):
    print(f"  {k} samples/question: SE {100 * math.sqrt(0.35 ** 2 / 200 + 0.12 / (200 * k)):.1f} points")
toks = 40 * 1000 * 4 * 4000
print(f"full suite {toks / 1e6:.0f}M tokens = {toks / tps / 3600:.1f} replica-hours = {8 * toks / tps / 3600:.0f} GPU-hours")

print("== E.15 coding agent")
inp, outp = 30 * 40e3, 30 * 500
no_cache = inp * IN / 1e6 + outp * OUT / 1e6
cache = 0.1 * inp * IN / 1e6 + 0.9 * inp * CACHED / 1e6 + outp * OUT / 1e6
print(f"input {inp / 1e6:.1f}M tokens; cost no cache ${no_cache:.2f}, with 90% cached ${cache:.2f}")
print(f"SE at 60%, 500 tasks: {100 * se(0.6, 500):.1f} points; pass^3 if independent: {0.6 ** 3:.3f}")
