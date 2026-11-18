import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
base = rng.normal(30, 8, 50000)   # basket value AED
print("same", round(psi(base, rng.normal(30, 8, 20000)), 4))
print("shift +2", round(psi(base, rng.normal(32, 8, 20000)), 4))
print("ramadan-like +6, wider", round(psi(base, rng.normal(36, 11, 20000)), 4))
print("ks p", stats.ks_2samp(base[:5000], rng.normal(32, 8, 5000)).pvalue)
# canary guardrail: error rate comparison
def two_prop(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); z = (x2 / n2 - x1 / n1) / se
    return z, 1 - stats.norm.cdf(z)
print(two_prop(120, 95000, 21, 5000))
