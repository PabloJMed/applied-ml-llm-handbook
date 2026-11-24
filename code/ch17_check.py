"""Chapter 17 · Pretraining at scale: runnable checks (NumPy, instant)."""
import math
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

# 17.11 worked budget for TP=8, PP=4, DP=32 on 1,024 GPUs (70B, 80 layers, h=8192, 64 heads, s=8192, micro-batch 1)
s, h, L, tp, pp, dp, m = 8192, 8192, 80, 8, 4, 32, 16
act_layer = s * h * 34 / tp / 1e9                      # attention scores recomputed (FlashAttention), sequence parallel
print("activations per layer %.3f GB; first stage, %d micro-batches in flight: %.1f GB" % (act_layer, pp, act_layer * L / pp * pp))
pg = 70e9 / tp / pp
print("params/GPU %.2fB; bf16 weights+grads %.1f GB; sharded Adam %.2f GB" % (pg / 1e9, pg * 4 / 1e9, pg * 12 / dp / 1e9))
step = 6 * 70e9 * 4 * 2**20 / (1024 * 989e12 * 0.40)
print("step time %.2f s; grad sync per GPU %.2f s at 50 GB/s; bubble %.1f%%, interleaved x2 %.1f%%; %.0f days for 15T"
      % (step, 2 * pg * 2 / 50e9, 100 * (pp - 1) / (m + pp - 1), 100 * (pp - 1) / (2 * m + pp - 1), 15e12 / (4 * 2**20) * step / 86400))

# ---------------------------------------------------------------------------------------------
# Additions: derivations and worked numbers added in the review pass
# ---------------------------------------------------------------------------------------------
print("\n--- additions ---")

# 17.2 where the 34 comes from (Korthikanti et al., bytes per token per layer in units of h, bf16, 1-byte dropout masks)
attn = {"LN output = QKV input": 2, "Q and K for QK^T": 4, "V": 2, "out-proj input": 2, "attn-out dropout mask": 1}
mlp = {"MLP input": 2, "first linear output (4h)": 8, "GeLU output (4h)": 8, "MLP dropout mask": 1}
ln = {"two layer-norm inputs": 4}
print("activation constant:", sum(attn.values()), "+", sum(mlp.values()), "+", sum(ln.values()), "=",
      sum(attn.values()) + sum(mlp.values()) + sum(ln.values()), "(x s*b*h bytes) + 5*a*s^2*b for the score matrices")
# a Llama-style block: no dropout, GQA (8 of 64 heads for K,V), SwiGLU with f = 3.5h (28,672 / 8,192)
f_over_h, kv_frac = 28672 / 8192, 8 / 64
attn_l = 2 + 2 + 2 * kv_frac + 2 * kv_frac + 2         # QKV input, Q, K, V, out-proj input
mlp_l = 2 + 4 * 2 * f_over_h                          # input + gate out, up out, silu(gate), product (each 2f bytes)
print(f"Llama-style block, rough: attention {attn_l:.1f} + SwiGLU {mlp_l:.1f} + norms 4 = {attn_l + mlp_l + 4:.1f} x s*b*h bytes")

# 17.3 collectives: alpha-beta costs, bytes sent per GPU for a buffer of M bytes on n GPUs
def coll_bytes(kind, M, n):
    return {"all-reduce": 2 * (n - 1) / n * M, "reduce-scatter": (n - 1) / n * M,
            "all-gather": (n - 1) / n * M, "all-to-all": (n - 1) / n * M}[kind]
M, n, bw = 140e9, 64, 50e9
ddp = coll_bytes("all-reduce", M, n)
z12 = coll_bytes("reduce-scatter", M, n) + coll_bytes("all-gather", M, n)
z3 = 2 * coll_bytes("all-gather", M, n) + coll_bytes("reduce-scatter", M, n)
print(f"70B, 64 GPUs, bf16 grads/weights 140 GB: DDP {ddp/1e9:.0f} GB/GPU ({ddp/bw:.1f} s), "
      f"ZeRO-1/2 {z12/1e9:.0f} GB ({z12/bw:.1f} s), ZeRO-3 {z3/1e9:.0f} GB ({z3/bw:.1f} s) = {z3/ddp:.2f}x DDP")
for tok_per_gpu in (4096, 8192, 65536):
    comp = 6 * 70e9 * tok_per_gpu * 64 / (64 * 989e12 * 0.40)
    print(f"  compute per step with {tok_per_gpu} tokens per GPU: {comp:.1f} s (gradient all-reduce {ddp/bw:.1f} s)")
# latency term: a ring all-reduce has 2(n-1) steps; with alpha = 10 us
for n_ in (8, 64, 1024):
    print(f"  ring all-reduce on {n_:4d} GPUs: {2*(n_-1)} latency steps = {2*(n_-1)*10e-6*1e3:.2f} ms at 10 us each")

# 17.4 TP traffic for the 70B layout: 4 all-reduces of b*s*h*2 bytes per layer per micro-batch (2 fwd, 2 bwd)
s, h, tp = 8192, 8192, 8
act_bytes = s * h * 2
per_layer = 4 * coll_bytes("all-reduce", act_bytes, tp)
per_step = per_layer * 20 * 16                       # 20 layers per stage, 16 micro-batches
print(f"TP: activation tensor {act_bytes/1e6:.0f} MB; per layer per micro-batch {per_layer/1e9:.2f} GB per GPU; "
      f"per step {per_step/1e9:.0f} GB: {per_step/450e9:.2f} s at 450 GB/s NVLink, {per_step/50e9:.1f} s at 50 GB/s network")

# 17.5/17.11 interleaved schedule memory and embedding imbalance
p, v = 4, 2
warm = (p - 0 - 1) * 2 + (v - 1) * p                  # Megatron warm-up micro-batch-chunks on rank 0
chunks_in_flight = warm + 1
layers_per_chunk = 80 // (p * v)
print(f"interleaved v={v}: rank 0 holds {chunks_in_flight} chunks x {layers_per_chunk} layers = "
      f"{chunks_in_flight*layers_per_chunk} layer-activations vs {p*80//p} for 1F1B "
      f"(ratio {chunks_in_flight*layers_per_chunk/(p*80//p):.3f}; 1+(p-1)/(p v) = {1+(p-1)/(p*v):.3f}); "
      f"{chunks_in_flight*layers_per_chunk*0.285:.1f} GB at 0.285 GB per layer")
V = 128256
layer_params = (70e9 - 2 * V * 8192) / 80
print(f"vocab {V}: embedding = LM head = {V*8192/1e9:.2f}B params; a layer {layer_params/1e9:.2f}B; "
      f"LM head matmul = {V*8192/layer_params:.2f} layers of FLOPs; logits per micro-batch fp32 / TP: {8192*V*4/8/1e9:.2f} GB")

# 17.8 NVL72-class layout: the same 70B on 16 racks x 72 = 1,152 Blackwell-class GPUs, TP=8, PP=1, DP=144
gpus, tp, dp = 1152, 8, 144
assert tp * dp == gpus
pg = 70e9 / tp
act = 80 * s * h * 34 / tp / 1e9
print(f"NVL72 layout TP=8 PP=1 DP=144: {pg/1e9:.2f}B params/GPU, bf16 w+g {pg*4/1e9:.1f} GB, "
      f"ZeRO-1 Adam {pg*12/dp/1e9:.2f} GB, activations 80 layers x 1 micro-batch {act:.1f} GB, "
      f"total {(pg*4 + pg*12/dp)/1e9 + act:.0f} GB")
batch = dp * 4 * 8192
peak_bw = 2.25e15
step_t = 6 * 70e9 * batch / (gpus * peak_bw * 0.40)
print(f"  global batch {dp} x 4 accum x 8192 = {batch/1e6:.2f}M tokens; step {step_t:.2f} s at 40% of 2.25 PFLOP/s; "
      f"days for 15T: {15e12/batch*step_t/86400:.0f}; grad all-reduce {2*(dp-1)/dp*pg*2/1e9:.1f} GB "
      f"= {2*(dp-1)/dp*pg*2/100e9:.2f} s at 100 GB/s")

# 17.10 MFU debugging walkthrough: target 4.35 s/step at 40% MFU; measured 25%
flops_gpu = 6 * 70e9 * 4 * 2**20 / 1024
print(f"model FLOPs per GPU per step {flops_gpu:.3e}; at peak {flops_gpu/989e12:.2f} s")
measured = flops_gpu / (989e12 * 0.25)
parts = {"GEMMs": 2.6, "attention kernels": 0.7, "small kernels (norms, elementwise, copies)": 1.2,
         "exposed communication": 1.0, "data-loader wait": 0.36}
non_bubble = sum(parts.values())
step_before = non_bubble / (1 - 3 / 19)
print(f"measured step at 25% MFU: {measured:.2f} s; trace parts sum {non_bubble:.2f} s + bubble -> {step_before:.2f} s "
      f"(bubble {step_before - non_bubble:.2f} s)")
after = dict(parts, **{"small kernels (norms, elementwise, copies)": 0.5, "exposed communication": 0.3, "data-loader wait": 0.0})
nb = sum(after.values()); step_after = nb / (1 - 3 / 35)
print(f"after fusion, overlap, prefetch and interleaving: {nb:.2f} s + bubble -> {step_after:.2f} s, "
      f"MFU {flops_gpu/(step_after*989e12):.1%}; GEMM-only efficiency {flops_gpu/(2.6*989e12):.0%} of peak")

# 17.12 fp8: E4M3 rounding, per-tensor vs per-block scaling with an outlier
def e4m3(x):
    """Round to the nearest FP8 E4M3 value (max 448, normals 2^-6..., subnormals k * 2^-9)."""
    x = np.asarray(x, float); s = np.sign(x); a = np.minimum(np.abs(x), 448.0)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -6)))
    step = 2.0 ** (e - 3)                                   # 3 mantissa bits
    return s * np.round(a / step) * step
vals = np.full(128, 0.01); vals[0] = 1000.0
scale_t = 1000.0 / 448
q = e4m3(vals / scale_t) * scale_t
print(f"per-tensor scale {scale_t:.3f}: 0.01 -> {q[1]:.5f} ({abs(q[1]-0.01)/0.01:.0%} error)")
block = vals[1:]; sb = np.abs(block).max() / 448
qb = e4m3(block / sb) * sb
print(f"per-block scale (block without the outlier): 0.01 -> {qb[0]:.5f} (error {abs(qb[0]-0.01)/0.01:.1%})")
rng8 = np.random.default_rng(0); xg = rng8.normal(0, 1, 100_000)
qg = e4m3(xg / (np.abs(xg).max() / 448)) * (np.abs(xg).max() / 448)
print(f"E4M3 relative RMS error on Gaussian values with absmax scaling: {np.sqrt(((qg-xg)**2).mean())/xg.std():.3f}")
step_bf16 = 6 * 70e9 * 4 * 2**20 / (1024 * 989e12 * 0.40)
for speedup in (1.3, 1.5):
    st = step_bf16 / speedup
    print(f"fp8 end-to-end speed-up {speedup}: step {st:.2f} s, {180/speedup:.0f} days; MFU vs bf16 peak "
          f"{0.40*speedup:.0%}, vs fp8 peak {0.40*speedup/2:.0%}")
# memory per parameter with fp8 matmuls: fp8 weight copy 1 + bf16 grad 2 + fp32 master 4 + bf16 Adam m, v 4
print("bytes/param: bf16 recipe 16; fp32 master + bf16 grads + bf16 Adam moments:", 4 + 2 + 2 + 2,
      "+ 1 for an fp8 weight copy (often transient)")

# 17.9 Young-Daly
print("Young-Daly interval sqrt(2 C M), C=1.2 min, M=180 min: %.1f min" % math.sqrt(2 * 1.2 * 180))

# 17.1 what 6ND leaves out: attention-score FLOPs (PaLM's 12 * L * d * s per token, forward + backward)
attn_flops = 12 * 80 * 8192 * 8192
print(f"attention FLOPs per token {attn_flops:.2e} vs 6N {6*70e9:.2e}: +{attn_flops/(6*70e9):.1%}")

# 17.6 ring attention overlap condition: s/n >= (F/B) * (h_kv/h)
for name, B in (("network 50 GB/s", 50e9), ("NVLink 450 GB/s", 450e9)):
    print(f"ring attention, 70B (h_kv/h = 1/8), {name}: needs >= {989e12/B/8:.0f} tokens per GPU to hide the KV transfer")

# 17.9 Young-Daly from waste(T) = C/T + T/(2M) (+ R/M per-failure restart cost)
for C in (1.2, 0.1):
    T = math.sqrt(2 * C * 180)
    print(f"C={C} min, M=180 min: T*={T:.1f} min, waste {C/T + T/360:.1%} (+ restart R=15 min: {15/180:.1%})")

# 17.8 NVL72 layout GPU-hours
batch = 144 * 4 * 8192; st = 6 * 70e9 * batch / (1152 * 2.25e15 * 0.40); days = 15e12 / batch * st / 86400
print(f"NVL72 layout: {days:.0f} days, {1152*days*24/1e6:.2f}M GPU-hours")

# interview: 7B memory
print("7B: Adam 16 B/param %.0f GB; FSDP over 8: %.0f GB/GPU; 8-bit Adam (2+2+4+2) %.0f GB; activations s=4096: %.1f GB"
      % (7e9 * 16 / 1e9, 7e9 * 16 / 8 / 1e9, 7e9 * 10 / 1e9, 32 * 4096 * 4096 * 34 / 1e9))
