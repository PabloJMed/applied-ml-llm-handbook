"""Chapter 17 · Pretraining at scale: runnable checks (NumPy, instant)."""
import numpy as np

# 17.1 the budget
N, D = 70e9, 15e12
flops = 6 * N * D
peak = 989e12          # dense bf16 FLOP/s per GPU, H100-class (round figure for arithmetic)
for gpus, mfu in ((1024, 0.40), (4096, 0.40), (16384, 0.38)):
    days = flops / (gpus * peak * mfu) / 86400
    print(f"70B x 15T tokens = {flops:.2e} FLOPs: {gpus:>5} GPUs at {mfu:.0%} MFU -> {days:5.0f} days, "
          f"{gpus * days * 24 / 1e6:.2f}M GPU-hours")

# 17.1 MFU from measured throughput
def mfu(tokens_per_s_per_gpu, n_params, n_layers, d_model, seq, peak=989e12):
    flops_per_token = 6 * n_params + 12 * n_layers * d_model * seq   # dense matmuls + attention scores
    return tokens_per_s_per_gpu * flops_per_token / peak
print("MFU at 900 tokens/s/GPU for 70B (80 layers, d=8192, seq 8192): %.1f%%" % (100 * mfu(900, 70e9, 80, 8192, 8192)))

# 17.2 memory per GPU: mixed precision with Adam = 16 bytes/param (2 weights + 2 grads + 12 fp32 master/m/v)
def per_gpu_gb(n_params, dp, stage):
    w, g, opt = 2 * n_params, 2 * n_params, 12 * n_params
    if stage >= 1: opt /= dp
    if stage >= 2: g /= dp
    if stage >= 3: w /= dp
    return (w + g + opt) / 1e9
for stage, name in ((0, "DDP"), (1, "ZeRO-1"), (2, "ZeRO-2"), (3, "ZeRO-3 / FSDP")):
    print(f"{name:14s} 70B over 64 GPUs: {per_gpu_gb(70e9, 64, stage):7.1f} GB/GPU (excluding activations)")

# 17.2 activation memory per layer (Korthikanti et al.), bytes, bf16, no tensor parallelism
def act_gb(seq, batch, d, heads, layers, recompute):
    per_layer = seq * batch * d * (34 + (0 if recompute else 5 * heads * seq / d))
    return per_layer * layers / 1e9
print("activations, 80 layers, d=8192, 64 heads, seq 8192, micro-batch 1: "
      f"{act_gb(8192, 1, 8192, 64, 80, False):.0f} GB full, {act_gb(8192, 1, 8192, 64, 80, True):.0f} GB "
      "with attention recomputed (flash attention does this implicitly)")

# 17.3 ring all-reduce = reduce-scatter + all-gather; each GPU sends 2(n-1)/n of the data
def ring_allreduce(chunks_per_gpu):
    n = len(chunks_per_gpu)
    data = [list(c) for c in chunks_per_gpu]           # data[g][k]: GPU g's copy of chunk k
    sent = 0
    for step in range(n - 1):                          # reduce-scatter
        msgs = [(g, (g - step) % n, data[g][(g - step) % n]) for g in range(n)]
        for g, k, val in msgs:
            data[(g + 1) % n][k] = data[(g + 1) % n][k] + val; sent += 1
    for step in range(n - 1):                          # all-gather
        msgs = [(g, (g + 1 - step) % n, data[g][(g + 1 - step) % n]) for g in range(n)]
        for g, k, val in msgs:
            data[(g + 1) % n][k] = val; sent += 1
    return data, sent
gpus = 4
grads = [np.arange(4) * (g + 1.0) for g in range(gpus)]       # each GPU's gradient, split into 4 chunks
out, sent = ring_allreduce(grads)
print("ring all-reduce result on every GPU:", [list(map(float, o)) for o in out][0],
      " expected:", list(map(float, sum(grads))), f" chunk-messages per GPU: {sent // gpus} = 2(n-1)")
size_gb, bw = 140, 50          # 70B bf16 gradients; ~50 GB/s effective per-GPU network bandwidth
print("all-reducing 140 GB of gradients over 64 GPUs at 50 GB/s: %.1f s per step (why overlap and sharding matter)"
      % (2 * (64 - 1) / 64 * size_gb / bw))

# 17.5 pipeline bubble
for p, m in ((8, 8), (8, 32), (16, 64)):
    print(f"pipeline stages {p}, micro-batches {m}: bubble {(p - 1) / (m + p - 1):.1%}")

# 17.8 a layout for 70B on 1,024 GPUs (8 per node)
tp, pp, dp = 8, 4, 32
assert tp * pp * dp == 1024
params_per_gpu = 70e9 / (tp * pp)
print(f"TP={tp} (inside a node), PP={pp}, DP={dp}: {params_per_gpu/1e9:.2f}B params per GPU; "
      f"weights+grads {params_per_gpu*4/1e9:.1f} GB, Adam states sharded over DP: {params_per_gpu*12/dp/1e9:.2f} GB")
global_batch_tokens = 4 * 2**20
micro, seq = 1, 8192
grad_accum = global_batch_tokens // (dp * micro * seq)
print(f"global batch 4M tokens = {dp} replicas x {micro} x {seq} tokens x {grad_accum} accumulation steps")

# 17.9 checkpoints and failures
ckpt_tb = 70e9 * 16 / 1e12
print(f"full checkpoint (bf16 weights + fp32 master + Adam): {ckpt_tb:.2f} TB")
gpu_mtbf_hours = 50_000
for n_gpus in (1024, 16384):
    print(f"{n_gpus:>5} GPUs with a per-GPU MTBF of {gpu_mtbf_hours:,} h: one failure every {gpu_mtbf_hours / n_gpus:.1f} h")
# expected lost work with checkpoint interval T and restart cost R, failure every M hours: ~ T/2 + R per failure
M, R = 3.0, 0.25
for T in (0.25, 0.5, 1, 2):
    waste = (T / 2 + R) / M + (0.02 / T)             # + checkpoint overhead of ~1.2 min each time (synchronous)
    print(f"checkpoint every {T:4.2f} h: ~{waste:.1%} of time lost")
