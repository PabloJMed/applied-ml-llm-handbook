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
