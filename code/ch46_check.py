"""Checks for the back-of-the-envelope numbers in Chapter 46 (ML system design)."""
import math

# --- Back-of-the-envelope calculator from the chapter ---
SECONDS_PER_DAY = 86_400
def qps(daily_users, requests_per_user, peak_factor=3.0):
    avg = daily_users * requests_per_user / SECONDS_PER_DAY
    return avg, avg * peak_factor
def embedding_memory_gb(n_rows, dim, bytes_per_value=4):
    return n_rows * dim * bytes_per_value / 1e9
def llm_cost_usd(n_calls, in_tokens, out_tokens, usd_per_m_in, usd_per_m_out, batch_discount=0.0):
    return n_calls * (in_tokens * usd_per_m_in + out_tokens * usd_per_m_out) / 1e6 * (1 - batch_discount)
avg, peak = qps(10e6, 5)
print(f"feed: {avg:,.0f} QPS average, {peak:,.0f} peak; items {embedding_memory_gb(1e6, 128):.2f} GB; "
      f"users {embedding_memory_gb(50e6, 128):.1f} GB; scorings/s {peak * 500:,.0f}")
print("catalogue: 20,000 restaurants x 50 dishes =", 20_000 * 50)
print(f"LLM job ${llm_cost_usd(10e6, 300, 50, 0.15, 0.60):,.0f}, batch ${llm_cost_usd(10e6, 300, 50, 0.15, 0.60, 0.5):,.0f}")

# --- Step 3: correcting probabilities after negative downsampling ---
def correct(q, w):
    """q: prediction of a model trained with negatives kept at rate w; returns the calibrated probability."""
    return q / (q + (1 - q) / w)
print("downsampling correction q=0.2, w=0.1 ->", round(correct(0.2, 0.1), 4),
      "; true base rate 2% becomes", round(0.02 / (0.02 + 0.98 * 0.1), 3), "in training")

# --- Research system design: 7B model, 2,000 GPUs, 30 days ---
gpus, days, eff = 2000, 30, 4e14
C = gpus * days * 86400 * eff
N = 7e9
D = C / (6 * N)
print(f"compute {C:.3g} FLOPs; MFU vs 989 TF bf16 dense = {eff / 989e12:.1%}; tokens {D:.3g}; tokens/param {D / N:,.0f}")
print(f"per-GPU throughput {eff / (6 * N):,.0f} tok/s; cluster {D / (days * 86400):.3g} tok/s")
N_opt = math.sqrt(C / (6 * 20)); print(f"Chinchilla-optimal (D = 20N): N = {N_opt:.3g}, D = {20 * N_opt:.3g}")
print("epochs over 15T unique tokens:", round(D / 15e12, 1))
# failure budget, scaled from Llama 3's 419 interruptions in 54 days on 16,384 GPUs
fail_per_day = 419 / 54 * gpus / 16384
mtbf_h = 24 / fail_per_day
ckpt_min, restart_min = 1.0, 10.0
tau = math.sqrt(2 * mtbf_h * 60 * ckpt_min)
lost_per_day = fail_per_day * (tau / 2 + restart_min) + (24 * 60 / tau) * ckpt_min
print(f"failures/day {fail_per_day:.2f}, MTBF {mtbf_h:.1f} h, Young/Daly interval {tau:.0f} min, "
      f"lost {lost_per_day:.0f} min/day = {lost_per_day / 1440:.1%}")
for ck in (0.25, 5.0):
    t2 = math.sqrt(2 * mtbf_h * 60 * ck)
    l2 = fail_per_day * (t2 / 2 + restart_min) + (24 * 60 / t2) * ck
    print(f"  checkpoint cost {ck} min -> interval {t2:.0f} min, loss {l2 / 1440:.1%}")

# --- LLM serving design: 70B dense model (80 layers, 8 KV heads of dim 128), bf16, H100 80 GB ---
layers, kv_heads, hd, bytes_ = 80, 8, 128, 2
kv_per_tok = 2 * layers * kv_heads * hd * bytes_
print(f"KV cache per token {kv_per_tok:,} bytes = {kv_per_tok / 1e6:.2f} MB")
weights_gb = 70e9 * 2 / 1e9
tp = 4; hbm = 80 * tp; free_gb = hbm - weights_gb - 0.1 * hbm   # keep ~10% for activations and overhead
print(f"TP={tp}: weights {weights_gb:.0f} GB, KV budget {free_gb:.0f} GB = {free_gb * 1e9 / kv_per_tok:,.0f} tokens"
      f" = {free_gb * 1e9 / kv_per_tok / 3000:,.0f} sequences of 3,000 tokens")
bw = 3.35e12 * tp
for batch in (16, 64, 128):
    t_w = weights_gb * 1e9 / bw
    t_kv = batch * 3000 * kv_per_tok / bw
    step = t_w + t_kv
    print(f"  batch {batch}: step {1e3 * step:.1f} ms (weights {1e3 * t_w:.1f}, KV {1e3 * t_kv:.1f}), "
          f"{batch / step:,.0f} tok/s per replica (ideal)")
# workload: 10M requests/day, 1,500 input and 300 output tokens, peak 2x average
req, tin, tout, peakf = 10e6, 1500, 300, 2
out_tps = req * tout / 86400; in_tps = req * tin / 86400
print(f"output {out_tps:,.0f} tok/s avg, {peakf * out_tps:,.0f} peak; input {in_tps:,.0f} tok/s avg")
prefill_flops = 2 * 70e9 * in_tps * peakf
prefill_gpus = prefill_flops / (989e12 * 0.5)
decode_replicas = peakf * out_tps / 2500           # ~2,500 tok/s per 4-GPU replica, about half the ideal at batch 64
print(f"peak prefill {prefill_flops:.3g} FLOP/s -> {prefill_gpus:.0f} GPUs at 50% MFU; "
      f"decode {decode_replicas:.0f} replicas = {decode_replicas * tp:.0f} GPUs")
ttft = 2 * 70e9 * tin / (tp * 989e12 * 0.5)
print(f"prefill compute for one 1,500-token prompt on {tp} GPUs at 50% MFU: {1e3 * ttft:.0f} ms")
total_gpus = round(prefill_gpus + decode_replicas * tp, -1); price = 2.5
day_cost = total_gpus * price * 24
print(f"{total_gpus:.0f} GPUs x ${price}/h = ${day_cost:,.0f}/day; per 1M output tokens ${day_cost / (req * tout / 1e6):.2f}; "
      f"per 1M total tokens ${day_cost / (req * (tin + tout) / 1e6):.2f}")
