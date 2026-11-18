import numpy as np
rng = np.random.default_rng(0)

def cosine(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(round(cosine(np.array([1., 2, 0]), np.array([2., 4, 0])), 3), round(cosine(np.array([1., 0]), np.array([0., 1])), 3))

# low-rank approximation
U0 = rng.normal(size=(100, 5)); V0 = rng.normal(size=(5, 80))
A = U0 @ V0 + 0.1 * rng.normal(size=(100, 80))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
for k in (1, 5, 20):
    Ak = U[:, :k] * S[:k] @ Vt[:k]
    print(k, round(np.linalg.norm(A - Ak) / np.linalg.norm(A), 3), "params", k * (100 + 80), "vs", 100 * 80)

# Bayes: fraud flag
prior, sens, fpr = 0.01, 0.95, 0.05
post = sens * prior / (sens * prior + fpr * (1 - prior))
print("P(fraud|flag)", round(post, 3))

# MLE vs MAP for a click-through rate
clicks, imps = 3, 10
mle = clicks / imps
a, b = 2, 38   # Beta prior with mean 5%
map_ = (clicks + a - 1) / (imps + a + b - 2)
post_mean = (clicks + a) / (imps + a + b)
print(round(mle, 3), round(map_, 3), round(post_mean, 3))

# gradient check
def f(w, X, y):
    p = 1 / (1 + np.exp(-X @ w)); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
def grad_f(w, X, y):
    p = 1 / (1 + np.exp(-X @ w)); return X.T @ (p - y) / len(y)
def numerical_grad(f, w, eps=1e-6, *args):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = eps
        g[i] = (f(w + e, *args) - f(w - e, *args)) / (2 * eps)
    return g
X = rng.normal(size=(50, 3)); y = (rng.random(50) < 0.5).astype(float); w = rng.normal(size=3)
print(np.allclose(grad_f(w, X, y), numerical_grad(f, w, 1e-6, X, y), atol=1e-6))

# entropy, cross-entropy, KL
def entropy(p): p = np.asarray(p); return -np.sum(p * np.log2(p, where=p > 0, out=np.zeros_like(p)))
def cross_entropy(p, q): return -np.sum(np.asarray(p) * np.log2(q))
def kl(p, q): return cross_entropy(p, q) - entropy(p)
p = np.array([0.7, 0.2, 0.1]); q = np.array([0.5, 0.3, 0.2])
print(round(entropy(p), 3), round(cross_entropy(p, q), 3), round(kl(p, q), 3), round(kl(q, p), 3))

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()
print(softmax(np.array([1000., 1001., 1002.])).round(3))
