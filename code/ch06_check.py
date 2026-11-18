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
