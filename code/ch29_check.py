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

# ---- 29.2 InfoNCE gradient worked example: dL/ds_ij = (p_ij - 1[j=i]) / tau
import math
s = np.array([0.8, 0.7, 0.3])
for tau in (0.05, 0.3):
    z = s / tau; p = np.exp(z - z.max()); p /= p.sum()
    g = p.copy(); g[0] -= 1; g /= tau
    print(tau, p.round(3), "dL/ds", g.round(2), "loss", round(-math.log(p[0]), 3))
# MI bound ceiling log B and false-negative probability 1-(1-f)^(B-1)
for B in (256, 1024, 4096, 32768): print(B, round(math.log(B), 2))
for f in (1e-4,):
    for B in (256, 4096, 32768): print("FN", B, round(1 - (1 - f) ** (B - 1), 3))
print('MarginMSE example', (0.1 - 7) ** 2)

# ---- 29.5/29.6 caveat: a skewed-spectrum, offset, rotated synthetic set (still synthetic)
rng = np.random.default_rng(1)
n, dim = 20000, 384
# skewed spectrum + common offset + clusters, randomly rotated (like a non-MRL real embedding)
scales = (np.arange(1, dim + 1) ** -0.5)
centers = rng.normal(size=(200, dim)) * scales
Z = centers[rng.integers(0, 200, n)] + 0.6 * rng.normal(size=(n, dim)) * scales
R, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
mu = rng.normal(size=dim); mu = 1.5*mu / np.linalg.norm(mu) * np.linalg.norm(Z, axis=1).mean()
X = Z @ R.T + mu
X /= np.linalg.norm(X, axis=1, keepdims=True); X = X.astype(np.float32)
Q = X[rng.choice(n, 100, replace=False)] + 0.02 * rng.normal(size=(100, dim)).astype(np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
truth = np.argsort(-(Q @ X.T), axis=1)[:, :10]
def recall(a): return np.mean([len(set(x) & set(t)) / 10 for x, t in zip(a, truth)])
def topk(Qm, Xm, k=10): return np.argsort(-(Qm @ Xm.T), axis=1)[:, :k]
def nrm(A): return A / np.linalg.norm(A, axis=1, keepdims=True)
print("mean cosine", float((X[:500] @ X[:500].T).mean()))
for k in (64, 128):
    print("trunc", k, recall(topk(nrm(Q[:, :k]), nrm(X[:, :k]))))
m = X.mean(0); U = np.linalg.svd(X - m, full_matrices=False)[2]
for k in (64, 128):
    P = U[:k].T
    print("pca", k, recall(topk(nrm((Q - m) @ P), nrm((X - m) @ P))))
def ham(Qb, Xb, k):
    return np.array([np.argsort(np.unpackbits(np.bitwise_xor(Xb, q), axis=1).sum(1), kind="stable")[:k] for q in Qb])
for name, c in (("raw", 0), ("centred", m)):
    Xb = np.packbits((X - c) > 0, axis=1); Qb = np.packbits((Q - c) > 0, axis=1)
    b = ham(Qb, Xb, 10); cand = ham(Qb, Xb, 100)
    rs = np.array([cc[np.argsort(-(X[cc] @ q))[:10]] for cc, q in zip(cand, Q)])
    print("binary", name, recall(b), "rescore100", recall(rs))
