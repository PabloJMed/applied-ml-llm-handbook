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

# 1.7 forward vs reverse KL: fit one Gaussian to a two-mode target
from scipy.optimize import minimize
xs = np.linspace(-8, 8, 4001); dx = xs[1] - xs[0]
def normal(x, m, s): return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
p = 0.5 * normal(xs, -2, 0.6) + 0.5 * normal(xs, 2, 0.6)
def kl(a, b): return np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx
fwd = minimize(lambda t: kl(p, normal(xs, t[0], np.exp(t[1]))), [0.5, 0.0]).x
rev = minimize(lambda t: kl(normal(xs, t[0], np.exp(t[1])), p), [1.0, -0.5]).x
print(f"forward KL(p||q) fit: mean {fwd[0]:.2f}, sd {np.exp(fwd[1]):.2f}  (covers both modes)")
print(f"reverse KL(q||p) fit: mean {rev[0]:.2f}, sd {np.exp(rev[1]):.2f}  (picks one mode)")

# 1.8 perplexity and bits: a model with loss 2.0 nats/token
loss = 2.0
print(f"loss {loss} nats/token = {loss/np.log(2):.2f} bits/token, perplexity {np.exp(loss):.2f}")

# ---- second-edition worked examples ---------------------------------------------------------
A2 = np.array([[3.0, 0.0], [4.0, 5.0]])
U2, S2_, Vt2 = np.linalg.svd(A2)
print("SVD of [[3,0],[4,5]]: singular values", S2_.round(3).tolist(), " rank-1 relative error",
      round(float(np.linalg.norm(A2 - S2_[0] * np.outer(U2[:, 0], Vt2[0])) / np.linalg.norm(A2)), 3))
d_in = d_out = 4096; r = 16
print(f"LoRA on a 4096x4096 matrix at rank {r}: {r*(d_in+d_out):,} trainable vs {d_in*d_out:,} ({r*(d_in+d_out)/(d_in*d_out):.2%})")
# Beta-binomial shrinkage: posterior mean (clicks + a) / (impressions + a + b) with a Beta(2, 38) prior
for clicks, imps in ((3, 10), (30, 100), (300, 1000), (0, 5)):
    print(f"{clicks}/{imps}: raw {clicks/imps:.3f}, smoothed {(clicks+2)/(imps+40):.3f}")
# backprop by hand: L = (w*x + b - y)^2 with x=2, w=3, b=1, y=5
x, w, b, y = 2.0, 3.0, 1.0, 5.0
z = w * x + b; Lval = (z - y) ** 2; dL_dz = 2 * (z - y)
print(f"forward z={z}, L={Lval}; dL/dz={dL_dz}, dL/dw={dL_dz*x}, dL/db={dL_dz}")
for pc in (0.9, 0.5, 0.1, 0.01):
    print(f"cross-entropy when the true class gets p={pc}: {-np.log(pc):.3f} nats")

# ---- review edition: derivations and new sections --------------------------------------------
# 1.1 unit vectors: ||a-b||^2 = 2 - 2cos; anisotropy and mean-centring
a_, b_ = np.array([0.6, 0.8]), np.array([1.0, 0.0])
print(f"||a-b||^2 = {np.sum((a_ - b_)**2):.2f}, 2 - 2cos = {2 - 2 * a_ @ b_:.2f}")
rng_a = np.random.default_rng(0); d = 768; mu = rng_a.normal(size=d)
E = rng_a.normal(size=(2000, d)) + 0.9 * np.sqrt(d) * mu / np.linalg.norm(mu)
def mean_cos(E):
    En = E / np.linalg.norm(E, axis=1, keepdims=True); return (En[:1000] @ En[1000:].T).mean()
print(f"anisotropic embeddings: mean cosine {mean_cos(E):.2f}, after mean-centring {mean_cos(E - E.mean(0)):.2f}")
# 1.2 singular values = sqrt(eig(A^T A)); PCA from the SVD of centred data
print("eig(A^T A) of [[3,0],[4,5]]:", np.linalg.eigvalsh(A2.T @ A2).round(3).tolist(), "cond", round(np.linalg.cond(A2), 3))
Z = np.random.default_rng(0).normal(size=(500, 3)) @ np.array([[2, 0, 0], [1, 1, 0], [0, 0, .3]])
Zc = Z - Z.mean(0); _, Sz, _ = np.linalg.svd(Zc, full_matrices=False)
print("PCA: S^2/(n-1)", (Sz**2 / 499).round(2).tolist(), " eig(cov)", np.linalg.eigvalsh(np.cov(Zc.T))[::-1].round(2).tolist())
# 1.4 one-weight MLE, ridge (Gaussian prior) and lasso (Laplace prior)
x3, y3 = np.array([1., 2, 3]), np.array([1., 3, 2]); Sxy, Sxx = x3 @ y3, x3 @ x3
sigma2, tau2, b_lap = 1.0, 0.5, 0.5
lam, lam1 = sigma2 / tau2, 2 * sigma2 / b_lap
from scipy.optimize import minimize_scalar
w_map = minimize_scalar(lambda w: np.sum((y3 - w * x3)**2) / (2 * sigma2) + w**2 / (2 * tau2)).x
print(f"MLE {Sxy/Sxx:.3f}; ridge lambda={lam:.0f}: {Sxy/(Sxx+lam):.4f} (numeric MAP {w_map:.4f}); lasso lambda1={lam1:.0f}: {(Sxy-lam1/2)/Sxx:.3f}")
# 1.7 softmax + cross-entropy gradient s - y, checked numerically
z3 = np.array([2., 1, 0]); y_oh = np.array([0., 1, 0]); s3 = softmax(z3)
ce = lambda z: -np.log(softmax(z)) @ y_oh
num = np.array([(ce(z3 + e) - ce(z3 - e)) / 2e-6 for e in np.eye(3) * 1e-6])
print("softmax", s3.round(3).tolist(), "grad s-y", (s3 - y_oh).round(3).tolist(), "numeric", num.round(3).tolist())
print("linear layer dL/dW = g x^T:", np.outer([1, -2], [3, 0.5]).tolist())
# 1.7 learning-rate limit 2/lambda on L = 0.5*10*w^2, and the condition number
for eta in (0.1, 0.15, 0.19, 0.21, 0.25):
    print(f"eta {eta}: factor {1 - eta*10:+.2f}, |w| after 20 steps {abs((1 - eta*10)**20):.3g}")
print(f"kappa=100 at eta=0.2: steps to shrink the flat direction 100x = {np.log(0.01)/np.log(0.98):.0f}; "
      f"at eta=0.19 = {np.log(0.01)/np.log(1-0.019):.0f}")
# 1.8 bits per byte
for name, L_tok, bytes_per_tok in (("A", 2.0, 4.0), ("B", 2.3, 4.8)):
    print(f"model {name}: {L_tok/np.log(2):.3f} bits/token, {L_tok/(np.log(2)*bytes_per_tok):.3f} bits/byte")
# 1.9 KL estimators k1, k3 on samples from p; importance-sampling effective sample size
p3, q3 = np.array([.7, .2, .1]), np.array([.5, .3, .2]); r3 = q3 / p3
k1, k3 = -np.log(r3), r3 - 1 - np.log(r3)
for name, k in (("k1", k1), ("k3", k3)):
    v = p3 @ k**2 - (p3 @ k)**2
    print(f"{name}: values {k.round(3).tolist()}, mean {p3 @ k:.4f}, per-sample var {v:.4f}, SE at n=100 {np.sqrt(v/100):.4f}")
print(f"true KL(p||q) = {np.sum(p3*np.log(p3/q3)):.4f} nats")
for qq in (q3, np.array([.6, .39, .01])):
    m2 = np.sum(p3**2 / qq); print(f"IS from q={qq.tolist()}: E_q[w^2] = {m2:.3f}, ESS per 100 = {100/m2:.0f}")
# 1.9 score-function estimator: two outcomes, with and without a baseline
for (r1, r0), base in (((1, 0), 0), ((1, 0), .5), ((10, 9), 0), ((10, 9), 9.5)):
    g1, g0 = (r1 - base) * 0.5, (r0 - base) * -0.5; m = (g1 + g0) / 2
    print(f"rewards {(r1, r0)}, baseline {base}: estimates {g1:+.2f}/{g0:+.2f}, mean {m:.2f}, var {(g1**2 + g0**2)/2 - m**2:.4f}")
# research exercise: 10-action softmax policy, per-sample gradient variance and cosine to the true gradient
K = 10; pi = np.full(K, 1 / K); rK = 5 + np.arange(K) / 10; true_g = pi * (rK - pi @ rK)
rng_r = np.random.default_rng(0)
for base in (0.0, pi @ rK):
    G = np.array([(rK[k] - base) * (np.eye(K)[k] - pi) for k in range(K)])
    var = pi @ np.sum((G - pi @ G)**2, 1)
    ests = np.array([((rK[a] - base)[:, None] * (np.eye(K)[a] - pi)).mean(0) for a in rng_r.integers(0, K, (1000, 64))])
    cos = np.mean(ests @ true_g / (np.linalg.norm(ests, axis=1) * np.linalg.norm(true_g)))
    print(f"baseline {base:.2f}: per-sample variance {var:.3f}, max bias {np.abs(ests.mean(0) - true_g).max():.3f}, mean cosine to true gradient {cos:.2f}")
# 1.10 Jensen and the ELBO on a two-state latent
print(f"Jensen: log E[X] = {np.log(2.5):.3f} >= E[log X] = {np.mean(np.log([1, 4])):.3f}")
pz, px_z = np.array([.5, .5]), np.array([.8, .2]); pxz = pz * px_z; post = pxz / pxz.sum(); qz = np.array([.5, .5])
elbo = np.sum(qz * np.log(pxz / qz))
print(f"log p(x) {np.log(pxz.sum()):.3f}, ELBO {elbo:.3f}, gap {np.log(pxz.sum()) - elbo:.3f}, KL(q||post) {np.sum(qz*np.log(qz/post)):.3f}")
# interview Q3 and Q5
for fpr_, rec_ in ((0.05, 0.95), (0.025, 0.95), (0.05, 0.99)):
    print(f"recall {rec_}, FPR {fpr_}: precision {rec_*0.01/(rec_*0.01 + fpr_*0.99):.3f}")
zq = -4.0; sg = 1 / (1 + np.exp(-zq))
print(f"z=-4, y=1: sigmoid {sg:.4f}, sigmoid' {sg*(1-sg):.4f}, MSE grad {(sg-1)*sg*(1-sg):.4f}, CE grad {sg-1:.3f}")
# practice 4: empirical-Bayes smoothing of CTRs
rng_c = np.random.default_rng(0); n_items = 1000
true_ctr = rng_c.beta(2, 38, n_items); imps = np.exp(rng_c.uniform(np.log(5), np.log(5000), n_items)).astype(int)
clicks_ = rng_c.binomial(imps, true_ctr); raw = clicks_ / imps
m_, v_ = raw.mean(), raw.var() - np.mean(raw.mean() * (1 - raw.mean()) / imps)
ab = m_ * (1 - m_) / v_ - 1; a_hat, b_hat = m_ * ab, (1 - m_) * ab
smooth = (clicks_ + a_hat) / (imps + a_hat + b_hat)
top_raw, top_s = np.argsort(-raw)[:10], np.argsort(-smooth)[:10]
print(f"fitted prior Beta({a_hat:.1f}, {b_hat:.1f}); raw top-10 with <20 impressions: {np.sum(imps[top_raw] < 20)}, "
      f"smoothed: {np.sum(imps[top_s] < 20)}; mean true CTR of raw top-10 {true_ctr[top_raw].mean():.3f}, smoothed {true_ctr[top_s].mean():.3f}")
