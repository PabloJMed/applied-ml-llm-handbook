import torch, torch.nn.functional as F, numpy as np
torch.manual_seed(0)

def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(1)
    return summed / mask.sum(1).clamp(min=1e-9)
te = torch.randn(2, 5, 8); am = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
print(torch.allclose(mean_pool(te, am)[0], te[0, :3].mean(0)))

def info_nce(q, d, temperature=0.05, symmetric=True):
    q, d = F.normalize(q, dim=-1), F.normalize(d, dim=-1)
    logits = q @ d.T / temperature
    labels = torch.arange(len(q))
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = (loss + F.cross_entropy(logits.T, labels)) / 2
    return loss
q = torch.randn(8, 16); d = q + 0.1 * torch.randn(8, 16)
print(round(info_nce(q, d).item(), 4), round(info_nce(q, torch.randn(8, 16)).item(), 4))

def matryoshka_loss(q, d, dims=(64, 128, 256, 768), temperature=0.05):
    return sum(info_nce(q[:, :k], d[:, :k], temperature) for k in dims) / len(dims)
q = torch.randn(8, 768); d = q + 0.3 * torch.randn(8, 768)
print(round(matryoshka_loss(q, d).item(), 4))

# quantisation of embeddings: int8 and binary with rescoring
rng = np.random.default_rng(0)
n, dim = 20000, 384
centers = rng.normal(size=(200, dim))
X = centers[rng.integers(0, 200, n)] + 0.6 * rng.normal(size=(n, dim))
X /= np.linalg.norm(X, axis=1, keepdims=True); X = X.astype(np.float32)
Q = X[rng.choice(n, 100, replace=False)] + 0.05 * rng.normal(size=(100, dim)).astype(np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
truth = np.argsort(-(Q @ X.T), axis=1)[:, :10]
def recall(approx): return np.mean([len(set(a) & set(t)) / 10 for a, t in zip(approx, truth)])

def int8_quantize(X):
    scale = np.abs(X).max(axis=0) / 127.0
    return np.round(X / scale).astype(np.int8), scale
Xq, scale = int8_quantize(X)
approx = np.argsort(-(Q @ (Xq.astype(np.float32) * scale).T), axis=1)[:, :10]
print("int8", recall(approx), "bytes", Xq.nbytes / X.nbytes)

Xb = np.packbits(X > 0, axis=1); Qb = np.packbits(Q > 0, axis=1)
def hamming_topk(qb, Xb, k):
    dist = np.unpackbits(np.bitwise_xor(Xb, qb), axis=1).sum(1)
    return np.argsort(dist, kind="stable")[:k]
approx_b = np.array([hamming_topk(qb, Xb, 10) for qb in Qb])
cand = [hamming_topk(qb, Xb, 100) for qb in Qb]
approx_rescored = np.array([c[np.argsort(-(X[c] @ q))[:10]] for c, q in zip(cand, Q)])
print("binary", recall(approx_b), "binary+rescore", recall(approx_rescored), "bytes", Xb.nbytes / X.nbytes)

# truncation without matryoshka training (random features): shows loss of recall
for k in [64, 128, 384]:
    Xk = X[:, :k] / np.linalg.norm(X[:, :k], axis=1, keepdims=True); Qk = Q[:, :k] / np.linalg.norm(Q[:, :k], axis=1, keepdims=True)
    print("trunc", k, recall(np.argsort(-(Qk @ Xk.T), axis=1)[:, :10]))
