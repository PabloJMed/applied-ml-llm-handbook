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
