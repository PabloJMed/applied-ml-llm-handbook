import numpy as np
rng = np.random.default_rng(0)

# spiral data, 3 classes
N, K = 100, 3
X = np.zeros((N * K, 2)); y = np.zeros(N * K, dtype=int)
for j in range(K):
    ix = range(N * j, N * (j + 1)); r = np.linspace(0.0, 1, N); t = np.linspace(j * 4, (j + 1) * 4, N) + rng.normal(0, 0.2, N)
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]; y[ix] = j

class MLP:
    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hidden)); self.b1 = np.zeros(d_hidden)      # He init for ReLU
        self.W2 = rng.normal(0, np.sqrt(2 / d_hidden), (d_hidden, d_out)); self.b2 = np.zeros(d_out)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h = np.maximum(0, self.z1)                       # ReLU
        logits = self.h @ self.W2 + self.b2
        logits -= logits.max(1, keepdims=True)
        self.p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return self.p

    def loss(self, y):
        return -np.mean(np.log(self.p[np.arange(len(y)), y] + 1e-12))

    def backward(self, y):
        n = len(y)
        dlogits = self.p.copy(); dlogits[np.arange(n), y] -= 1; dlogits /= n     # softmax + CE
        self.dW2 = self.h.T @ dlogits; self.db2 = dlogits.sum(0)
        dh = dlogits @ self.W2.T
        dz1 = dh * (self.z1 > 0)                              # ReLU derivative
        self.dW1 = self.X.T @ dz1; self.db1 = dz1.sum(0)

    def step(self, lr):
        for p, g in [(self.W1, self.dW1), (self.b1, self.db1), (self.W2, self.dW2), (self.b2, self.db2)]:
            p -= lr * g

m = MLP(2, 64, 3)
for epoch in range(3000):
    m.forward(X); l = m.loss(y); m.backward(y); m.step(1.0)
acc = (m.forward(X).argmax(1) == y).mean()
print("loss", round(l, 3), "acc", round(acc, 3))

# gradient check on W1 entry
m2 = MLP(2, 8, 3, seed=1); m2.forward(X[:20]); m2.backward(y[:20])
i, j, eps = 0, 3, 1e-5
orig = m2.W1[i, j]
m2.W1[i, j] = orig + eps; m2.forward(X[:20]); lp = m2.loss(y[:20])
m2.W1[i, j] = orig - eps; m2.forward(X[:20]); lm = m2.loss(y[:20])
m2.W1[i, j] = orig
print("grad check", np.isclose((lp - lm) / (2 * eps), m2.dW1[i, j], atol=1e-6))

# linear baseline
from sklearn.linear_model import LogisticRegression
print("logreg acc", LogisticRegression().fit(X, y).score(X, y))

# init experiment: activation std through 20 layers
def activation_stats(init, layers=20, d=256):
    h = rng.normal(size=(500, d))
    for _ in range(layers):
        W = rng.normal(0, init(d), (d, d)); h = np.maximum(0, h @ W)
    return h.std()
print("std after 20 layers: small", activation_stats(lambda d: 0.01), "he", round(activation_stats(lambda d: np.sqrt(2 / d)), 3), "large", activation_stats(lambda d: 1.0))

def dropout(h, p, train=True, rng=rng):
    if not train or p == 0: return h
    mask = (rng.random(h.shape) > p) / (1 - p)     # inverted dropout: scale at train time
    return h * mask
h = np.ones((10000,)); print(round(dropout(h, 0.3).mean(), 3))

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(0); var = x.var(0)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(1, keepdims=True); var = x.var(1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
xx = rng.normal(5, 3, (4, 6))
print(np.round(batchnorm_forward(xx, 1, 0).mean(0), 6)[:3], np.round(layernorm_forward(xx, 1, 0).mean(1), 6)[:3])

# ---- review-improvements worked examples ----------------------------------------------------
# 5.2 softmax + cross-entropy gradient: analytic p - y vs central differences
z = np.array([2.0, 1.0, 0.0]); p3 = np.exp(z) / np.exp(z).sum()
L3 = lambda zz: -np.log(np.exp(zz[0]) / np.exp(zz).sum())
num = [(L3(z + 1e-6 * np.eye(3)[j]) - L3(z - 1e-6 * np.eye(3)[j])) / 2e-6 for j in range(3)]
print("softmax-CE p", np.round(p3, 3), "grad", np.round(p3 - np.eye(3)[0], 3), "numerical", np.round(num, 6))

# 5.3 dead ReLUs against learning rate (64-unit MLP, 1,000 full-batch steps, 5 seeds)
for lr in (0.3, 1, 3, 10, 30):
    dead, accs = [], []
    for seed in range(5):
        mm = MLP(2, 64, 3, seed=seed)
        for _ in range(1000):
            mm.forward(X); mm.backward(y); mm.step(lr)
        mm.forward(X); dead.append((mm.z1 <= 0).all(0).mean()); accs.append((mm.p.argmax(1) == y).mean())
    print(f"lr {lr}: dead fraction {np.mean(dead):.3f} (per seed {np.round(dead, 2)}), accuracy {np.mean(accs):.3f}")

# 5.4 the init experiment exactly as drawn in tools/figures.py (same generator sequence)
rf = np.random.default_rng(0)
for _ in range(3): rf.normal(0, 0.2, 200)                   # the spiral noise drawn first in figures.py
xin = rf.standard_normal((512, 256))
for name, sd in (("0.01", lambda d: 0.01), ("1.0", lambda d: 1.0), ("He", lambda d: np.sqrt(2 / d))):
    hh = xin
    for _ in range(20): hh = np.maximum(0, hh @ rf.normal(0, sd(256), (256, 256)))
    print(f"figure run, init {name}: std after 20 layers {hh.std():.2e}")
print(f"per-layer scale factor, std 0.01 width 256: {np.sqrt(256 * 1e-4 / 2):.3f}; after 20 layers {np.sqrt(256 * 1e-4 / 2) ** 20:.1e}")

# 5.6 number formats
try:
    import torch
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        fi = torch.finfo(dt); print(dt, "max", fi.max, "eps", fi.eps, "smallest normal", fi.smallest_normal)
    print("fp16(1e-8) =", torch.tensor(1e-8, dtype=torch.float16).item(), "; fp16(70000) =", torch.tensor(70000.0, dtype=torch.float16).item(),
          "; bf16(1 + 1/300) =", torch.tensor(1 + 1 / 300, dtype=torch.bfloat16).item())
except ImportError:
    print("torch not installed: format limits not checked")

# 5.8 residual connections: gradient ratio input/output through 20 ReLU layers of width 256
print("scalar: plain 0.5^20 =", 0.5 ** 20, "; residual 1.5^20 =", round(1.5 ** 20), "; (1 + 1/40)^20 =", round((1 + 1 / 40) ** 20, 2))
def grad_ratio(residual, std_scale, L=20, d=256, seed=0, branch=1.0):
    rr = np.random.default_rng(seed)
    x = rr.standard_normal((256, d)); Ws = [rr.normal(0, std_scale * np.sqrt(2 / d), (d, d)) for _ in range(L)]
    hcur, zs = x, []
    for W in Ws:
        zz = hcur @ W; zs.append(zz); f = np.maximum(0, zz) * branch
        hcur = hcur + f if residual else f
    g = rr.standard_normal(hcur.shape) / np.sqrt(hcur.size); g_top = np.linalg.norm(g)
    for W, zz in zip(reversed(Ws), reversed(zs)):
        gb = (g * branch * (zz > 0)) @ W.T
        g = g + gb if residual else gb
    return np.linalg.norm(g) / g_top, hcur.std()
for label, args in (("plain, He", (False, 1.0)), ("plain, half He", (False, 0.5)),
                    ("residual, half He", (True, 0.5)), ("residual, He", (True, 1.0))):
    print(f"{label}: grad ratio {grad_ratio(*args)[0]:.3g}, output std {grad_ratio(*args)[1]:.3g}")
print(f"residual, He, branches / sqrt(2L): grad ratio {grad_ratio(True, 1.0, branch=1 / np.sqrt(40))[0]:.3g}, "
      f"output std {grad_ratio(True, 1.0, branch=1 / np.sqrt(40))[1]:.3g}")
