import re, string, math
from collections import Counter
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))

def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)

print(exact_match("The 24 hours.", "24 hours"), round(token_f1("within 24 hours of delivery", "24 hours"), 2))

def cohens_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in labels)
    return (po - pe) / (1 - pe)

human = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
judge = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
print(round(cohens_kappa(human, judge), 3), round(cohen_kappa_score(human, judge), 3))

def judge_tpr_tnr(human, judge):
    h, j = np.asarray(human), np.asarray(judge)
    return np.mean(j[h == 1] == 1), np.mean(j[h == 0] == 0)
tpr, tnr = judge_tpr_tnr(human, judge)
print(tpr, tnr)

def corrected_pass_rate(observed_rate, tpr, tnr):
    """Rogan-Gladen: true rate from a judge's observed rate and its error rates."""
    return min(max((observed_rate + tnr - 1) / (tpr + tnr - 1), 0.0), 1.0)
print(round(corrected_pass_rate(0.80, tpr, tnr), 3))

def wilson_interval(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half
print([round(x, 3) for x in wilson_interval(85, 100)], [round(x, 3) for x in wilson_interval(850, 1000)])

def mcnemar_exact(a_correct, b_correct):
    a, b = np.asarray(a_correct), np.asarray(b_correct)
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = stats.binomtest(min(n01, n10), n, 0.5).pvalue if n else 1.0
    return n01, n10, p
rng = np.random.default_rng(0)
A = rng.random(300) < 0.80
B = A.copy(); flip = rng.random(300)
B[(~A) & (flip < 0.35)] = True; B[A & (flip < 0.05)] = False
print(mcnemar_exact(A, B), A.mean(), B.mean())

def n_for_two_proportions(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)
print(n_for_two_proportions(0.80, 0.85), n_for_two_proportions(0.80, 0.82))

# code-based checks
import json
def run_checks(output, checks):
    results = {}
    for name, fn in checks.items():
        try:
            results[name] = bool(fn(output))
        except Exception:
            results[name] = False
    return results
checks = {
    "valid_json": lambda o: isinstance(json.loads(o), dict),
    "has_answer": lambda o: len(json.loads(o)["answer"]) > 0,
    "under_80_words": lambda o: len(json.loads(o)["answer"].split()) <= 80,
    "cites_source": lambda o: len(json.loads(o)["sources"]) > 0,
    "no_email": lambda o: re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", o) is None,
}
print(run_checks('{"answer": "You can request a voucher within 24 hours.", "sources": ["policy-17#1"]}', checks))

# 24.11 worked calibration: 200 expert labels (150 pass, 50 fail)
import math
tp, fn, tn, fp = 140, 10, 33, 17
human = [1] * (tp + fn) + [0] * (tn + fp); judge = [1] * tp + [0] * fn + [0] * tn + [1] * fp
tpr, tnr = tp / (tp + fn), tn / (tn + fp)
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)
po = (tp + tn) / 200; pe = (150 / 200) * ((tp + fp) / 200) + (50 / 200) * ((tn + fn) / 200)
print(f"agreement {po:.3f}, kappa {(po - pe) / (1 - pe):.3f}, TPR {tpr:.3f} {wilson(tp, tp + fn)}, TNR {tnr:.3f} {wilson(tn, tn + fp)}")
print("observed 80%% -> corrected %.3f; true gap 15 points looks like %.1f" % ((0.80 + tnr - 1) / (tpr + tnr - 1), 15 * (tpr + tnr - 1)))

# ---------------------------------------------------------------------------
# 24.7 Rogan-Gladen derivation check and propagation of judge uncertainty (parametric bootstrap
# over the calibration counts and the production sample)
tpr_hat, tnr_hat = 140 / 150, 33 / 50
J = tpr_hat + tnr_hat - 1
print(f"Youden J = {J:.3f}; variance inflation 1/J^2 = {1 / J**2:.2f}")
print("corrected at TNR interval ends:", [round(corrected_pass_rate(0.80, tpr_hat, t), 3) for t in (0.522, 0.776)])

def corrected_interval(k_prod, n_prod, tp, n_pos, tn, n_neg, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.binomial(n_prod, k_prod / n_prod, B) / n_prod
    tpr_b = rng.binomial(n_pos, tp / n_pos, B) / n_pos
    tnr_b = rng.binomial(n_neg, tn / n_neg, B) / n_neg
    est = np.clip((obs + tnr_b - 1) / (tpr_b + tnr_b - 1), 0, 1)
    return np.percentile(est, [2.5, 97.5])
lo, hi = corrected_interval(800, 1000, 140, 150, 33, 50)
print(f"corrected pass rate 0.775, 95% bootstrap interval [{lo:.3f}, {hi:.3f}] "
      f"(production sample alone: +/- {1.96 * math.sqrt(0.8 * 0.2 / 1000):.3f})")
lo2, hi2 = corrected_interval(800, 1000, 140, 150, 330, 500)
print(f"same judge calibrated on 500 failures instead of 50: [{lo2:.3f}, {hi2:.3f}]")

# 24.8 sample-size formulas
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
print(f"(za+zb)^2 = {(za + zb)**2:.2f}; unpaired 80->85: {(za + zb)**2 * (0.8 * 0.2 + 0.85 * 0.15) / 0.05**2:.0f} per arm; "
      f"paired with 10% discordant: {(za + zb)**2 * 0.10 / 0.05**2:.0f} cases")

# 24.8 several samples per case: Var = (sigma_q^2 + sigma_w^2 / K) / n with per-case p ~ Beta(1.4, 0.6)
a_, b_ = 1.4, 0.6
mean_p = a_ / (a_ + b_); var_q = a_ * b_ / ((a_ + b_) ** 2 * (a_ + b_ + 1)); var_w = mean_p * (1 - mean_p) - var_q
print(f"mean {mean_p:.2f}, between-case variance {var_q:.3f}, within-case variance {var_w:.3f}")
for K in (1, 2, 4, 8, 10**9):
    print(f"  n=200, K={K}: SE {math.sqrt((var_q + var_w / K) / 200):.4f}")
p_sim = rng.beta(a_, b_, 200)                         # simulate to confirm the formula
for K in (1, 4):
    means = [rng.binomial(K, p_sim).mean() / K for _ in range(4000)]
    print(f"  simulated (fixed 200 cases, sampling noise only) K={K}: SD {np.std(means):.4f}")
# flaky-test gate: pass if >= 3 of 4 samples pass
for p in (0.95, 0.9, 0.5):
    print(f"  case with p={p}: single-run pass {p:.2f}, P(>=3 of 4) {stats.binom.sf(2, 4, p):.3f}")
lo3, hi3 = corrected_interval(800, 1000, 1400, 1500, 330, 500)
q_good = 0.9333 * 0.775 + 0.10 * 0.225                     # same true rate, judge with TNR 0.90
lo4, hi4 = corrected_interval(round(q_good * 1000), 1000, 140, 150, 45, 50)
print(f"tenfold calibration set: [{lo3:.3f}, {hi3:.3f}]; TNR 0.90 judge, 200 labels: [{lo4:.3f}, {hi4:.3f}]")
print(f"P(2% failure mode unseen in 100 traces) = {0.98 ** 100:.3f}")
