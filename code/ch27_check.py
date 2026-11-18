"""Chapter 27 · Alignment and safety research: runnable checks (NumPy + SciPy + scikit-learn, seconds)."""
import numpy as np
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# 27.2 specification gaming: a "solution" that special-cases the visible tests
visible = [((2, 3), 5), ((10, -4), 6), ((0, 0), 0)]
hidden = [((7, 8), 15), ((-5, -5), -10), ((100, 1), 101), ((3, 3), 6)]
def honest(a, b): return a + b
def gamed(a, b):
    table = {(2, 3): 5, (10, -4): 6, (0, 0): 0}            # memorises the tests the reward checks
    return table.get((a, b), 0)
def pass_rate(f, tests): return np.mean([f(*x) == y for x, y in tests])
for name, f in (("honest", honest), ("gamed", gamed)):
    print(f"{name:7s}: visible tests {pass_rate(f, visible):.2f}, hidden tests {pass_rate(f, hidden):.2f}")

# 27.4 attack success rate with a confidence interval
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
for k, n in ((12, 200), (3, 200), (0, 200)):
    lo, hi = wilson(k, n)
    print(f"jailbreak succeeded {k}/{n}: ASR {k/n:.1%}, 95% CI [{lo:.1%}, {hi:.1%}]")

# 27.9 absence of evidence: 0 successes in n trials, upper bound on the true success rate
for n in (10, 50, 300):
    upper = 1 - 0.05 ** (1 / n)                               # exact one-sided 95% bound
    print(f"0/{n} successes: true rate could still be up to {upper:.1%} (rule of three: {3/n:.1%})")
print("a capability with a 5%% per-attempt success rate, tried 50 times: P(at least one success) = %.3f" % (1 - 0.95 ** 50))

# 27.6 weak-to-strong generalisation: a strong model trained only on a weak supervisor's labels
from sklearn.neighbors import KNeighborsClassifier
X, y = make_classification(n_samples=24000, n_features=20, n_informative=8, n_redundant=2,
                           n_clusters_per_class=3, class_sep=1.2, flip_y=0.0, random_state=0)
Xs, ys, Xt, yt = X[4000:16000], y[4000:16000], X[16000:], y[16000:]
ceiling = (HistGradientBoostingClassifier(random_state=0).fit(Xs, ys).predict(Xt) == yt).mean()
for wname, weak in (("linear (systematic errors)", LogisticRegression(max_iter=2000)),
                    ("1-NN on 300 examples (scattered errors)", KNeighborsClassifier(1))):
    weak.fit(X[:300], y[:300])
    weak_labels, a_weak = weak.predict(Xs), (weak.predict(Xt) == yt).mean()
    for iters in (30, 300):
        strong = HistGradientBoostingClassifier(random_state=0, max_iter=iters, max_leaf_nodes=15).fit(Xs, weak_labels)
        a = (strong.predict(Xt) == yt).mean()
        print(f"weak = {wname:40s} {a_weak:.3f}; strong on weak labels ({iters:>3} rounds) {a:.3f}; "
              f"ceiling {ceiling:.3f}; PGR {(a - a_weak) / (ceiling - a_weak):.0%}")

# 27.3 sycophancy metric: answer flips under user pushback, with a paired test
before = rng.random(300) < 0.80                                  # correct before pushback
flip_prob = np.where(before, 0.22, 0.10)                          # pushback: "I think that's wrong"
after = np.where(rng.random(300) < flip_prob, ~before, before)
c2w = int((before & ~after).sum()); w2c = int((~before & after).sum())
print(f"accuracy before pushback {before.mean():.3f}, after {after.mean():.3f}; correct->wrong {c2w}, "
      f"wrong->correct {w2c}; McNemar p = {stats.binomtest(c2w, c2w + w2c, 0.5).pvalue:.2g}")
