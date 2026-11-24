import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.decomposition import LatentDirichletAllocation

texts = [
    "pizzeria napoli margherita calzone", "burger house cheeseburger fries", "sushi tokyo maki nigiri",
    "farmacia central ibuprofeno paracetamol", "supermercado dia leche pan huevos", "kebab istanbul doner durum",
    "pizza express pepperoni four cheese", "smash burgers bacon double", "ramen bar tonkotsu gyoza",
    "parafarmacia crema solar vitaminas", "fruteria manzanas platanos", "pharmacy plus aspirin bandages",
]
labels = ["restaurant", "restaurant", "restaurant", "pharmacy", "grocery", "restaurant",
          "restaurant", "restaurant", "restaurant", "pharmacy", "grocery", "pharmacy"]
# Evaluate on names the model has never seen. (Duplicating the list before a random split
# would put copies of training rows in the test set: the leakage Chapter 3.7 warns about.)
X_tr, y_tr = texts, labels
X_te = ["pizzeria roma diavola", "burger bros onion rings", "farmacia norte aspirina",
        "supermercado eroski arroz aceite", "thai garden pad thai", "pharmacy express vitamins"]
y_te = ["restaurant", "restaurant", "pharmacy", "grocery", "restaurant", "pharmacy"]

clf = make_pipeline(
    make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True),
    ),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
clf.fit(X_tr, y_tr)
print(f1_score(y_te, clf.predict(X_te), average="macro"))
print(clf.predict(["farmasi ibuprofen", "pizeria roma"]))
proba = clf.predict_proba(["random shop name"])
print(proba.max())

# BIO decoding + entity F1
def bio_to_spans(tags):
    spans, start, label = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != label):
            if start is not None:
                spans.append((start, i, label))
                start, label = None, None
            if tag.startswith("B-") or tag.startswith("I-"):
                start, label = i, tag[2:]
    return set(spans)

def entity_f1(true_tags, pred_tags):
    tp = fp = fn = 0
    for t, p in zip(true_tags, pred_tags):
        ts, ps = bio_to_spans(t), bio_to_spans(p)
        tp += len(ts & ps); fp += len(ps - ts); fn += len(ts - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

tokens = ["Spicy", "chicken", "burger", "with", "vegan", "mayo", "330", "ml"]
true = [["B-ATTR", "B-ING", "B-DISH", "O", "B-DIET", "B-ING", "B-SIZE", "I-SIZE"]]
pred = [["B-ATTR", "B-DISH", "I-DISH", "O", "B-DIET", "B-ING", "B-SIZE", "I-SIZE"]]
print(bio_to_spans(true[0]))
print(entity_f1(true, pred))

# LDA
reviews = ["cold food arrived late", "courier was rude and late", "pizza was cold", "great sushi fresh fish",
           "delivery late again", "fresh tasty sushi rolls", "packaging leaked sauce everywhere",
           "sauce spilled bag wet", "rider friendly fast", "fast delivery hot food"] * 5
cv = CountVectorizer(stop_words="english")
Xc = cv.fit_transform(reviews)
lda = LatentDirichletAllocation(n_components=3, random_state=0).fit(Xc)
vocab = np.array(cv.get_feature_names_out())
for k, comp in enumerate(lda.components_):
    print(k, vocab[np.argsort(-comp)[:5]])

# class-based TF-IDF for cluster labels (BERTopic idea)
def c_tf_idf(docs, cluster_ids, top_n=5):
    df = pd.DataFrame({"doc": docs, "c": cluster_ids})
    joined = df.groupby("c")["doc"].apply(" ".join)        # one big document per cluster
    cv = CountVectorizer(stop_words="english")
    counts = cv.fit_transform(joined).toarray().astype(float)
    tf = counts / counts.sum(axis=1, keepdims=True)
    avg_words = counts.sum() / len(joined)
    idf = np.log(1 + avg_words / counts.sum(axis=0))
    scores = tf * idf
    vocab = cv.get_feature_names_out()
    return {c: [vocab[i] for i in np.argsort(-row)[:top_n]] for c, row in zip(joined.index, scores)}
clusters = [0, 0, 0, 1, 0, 1, 2, 2, 3, 3] * 5
print(c_tf_idf(reviews, clusters))

# ROUGE
def rouge_n(candidate, reference, n=1):
    grams = lambda t: Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    c, r = grams(candidate.lower().split()), grams(reference.lower().split())
    overlap = sum((c & r).values())
    prec = overlap / max(sum(c.values()), 1)
    rec = overlap / max(sum(r.values()), 1)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l(candidate, reference):
    c, r = candidate.lower().split(), reference.lower().split()
    lcs = lcs_len(c, r)
    prec, rec = lcs / max(len(c), 1), lcs / max(len(r), 1)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

ref = "customers praise the fresh sushi but complain about late delivery"
good = "reviews like the fresh sushi but say delivery is often late"
wrong = "customers praise the late delivery but complain about the fresh sushi"
for name, cand in [("faithful paraphrase", good), ("meaning reversed", wrong)]:
    print(name, round(rouge_n(cand, ref, 1), 2), round(rouge_n(cand, ref, 2), 2), round(rouge_l(cand, ref), 2))

# 41.10 entity-level evaluation: exact vs partial span matching
def spans(text, items):
    """(type, phrase) -> (type, start, end) character spans, found left to right."""
    out = []
    for typ, phrase in items:
        start = text.index(phrase); out.append((typ, start, start + len(phrase)))
    return out

def span_f1(gold, pred, mode="exact"):
    """mode 'exact': same type and boundaries; 'partial': same type and any overlap (each gold used once)."""
    used, tp = set(), 0
    for t, s, e in pred:
        for k, (gt, gs, ge) in enumerate(gold):
            if k in used or t != gt:
                continue
            if (mode == "exact" and (s, e) == (gs, ge)) or (mode == "partial" and s < ge and gs < e):
                used.add(k); tp += 1; break
    p, r = tp / len(pred), tp / len(gold)
    return round(p, 2), round(r, 2), round(2 * p * r / (p + r) if p + r else 0.0, 2)

text = "Two shawarma plates and a large pepsi from Al Safadi in Dubai Marina, no onions"
gold = spans(text, [("DISH", "shawarma plates"), ("DISH", "large pepsi"), ("RESTAURANT", "Al Safadi"), ("AREA", "Dubai Marina")])
pred = spans(text, [("DISH", "shawarma"), ("DISH", "large pepsi"), ("RESTAURANT", "Al Safadi"), ("AREA", "Marina"), ("DISH", "onions")])
print("exact", span_f1(gold, pred, "exact"), "partial", span_f1(gold, pred, "partial"))

# ---------------------------------------------------------------------------
# 41.1 / 41.11 cost arithmetic
print("\n--- cost arithmetic ---")
p_in, p_out, gpu_month = 0.15e-6, 0.60e-6, 1.0 * 730
short = 400 * p_in + 10 * p_out
print("400-token call", short, "monthly at 600k/day", round(600e3 * 30 * short), "batch", round(600e3 * 30 * short / 2))
print("break-even real time", round(gpu_month / (30 * short)), "batch", round(gpu_month / (30 * short / 2)))
for P in (4000, 5000, 6000):
    call = P * p_in * 0.10 + 300 * p_in + 10 * p_out          # cached prefix at 10% of the input price
    nocache = (P + 300) * p_in + 10 * p_out
    print(f"prefix {P}: per call {call:.3e}; month real time {600e3*30*call:.0f}, batch {600e3*30*call/2:.0f}; "
          f"batch break-even {gpu_month/(30*call/2):.0f}/day; no cache batch {600e3*30*nocache/2:.0f}; "
          f"label 200k in batch {200e3*call/2:.1f}")
print("95% half-width of 92% on 1,000 items", round(1.96 * np.sqrt(0.92 * 0.08 / 1000), 4),
      "90% on 500 items", round(1.96 * np.sqrt(0.9 * 0.1 / 500), 4))
print("Cohen kappa 0.85 observed, 0.40 chance:", round((0.85 - 0.40) / (1 - 0.40), 2))
print("share of 8-claim summaries with an unsupported claim at 2% per claim", round(1 - 0.98 ** 8, 3))

# c-TF-IDF walk-through for "late" and "food" in cluster 0
df_ = pd.DataFrame({"doc": reviews, "c": clusters}).groupby("c")["doc"].apply(" ".join)
cv_ = CountVectorizer(stop_words="english"); cnt = cv_.fit_transform(df_).toarray().astype(float)
voc = list(cv_.get_feature_names_out()); A = cnt.sum() / len(df_)
for w_ in ("late", "food"):
    j = voc.index(w_); tf_ = cnt[0, j] / cnt[0].sum(); idf_ = np.log(1 + A / cnt[:, j].sum())
    print(f"c-TF-IDF {w_}: words per cluster {cnt.sum(1)}, A={A}, tf={tf_:.3f}, idf={idf_:.3f}, weight={tf_*idf_:.3f}")

# BLEU clipped precisions
def ngrams(t, n): return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
cand, refb = "the cat sat on the mat".split(), "the cat is on the mat".split()
print("BLEU clipped precisions", [f"{sum((ngrams(cand, n) & ngrams(refb, n)).values())}/{sum(ngrams(cand, n).values())}" for n in (1, 2, 3, 4)],
      "BLEU-2", round(np.sqrt(5 / 6 * 3 / 5), 3), "clipping 'the the the the' vs 'the cat':",
      sum((ngrams("the the the the".split(), 1) & ngrams("the cat".split(), 1)).values()), "/ 4")

# subword label alignment
def align_labels(word_labels, word_ids):
    out, prev = [], None
    for w in word_ids:
        out.append(-100 if (w is None or w == prev) else word_labels[w]); prev = w
    return out
print("align", align_labels([3, 0, 5], [None, 0, 0, 0, 1, 2, 2, None]))

# 41.2 research exercise: class weights break calibration; fixes
from scipy.optimize import minimize_scalar
from sklearn.datasets import make_classification
def ece(P, y, bins=10):
    conf, acc = P.max(1), P.argmax(1) == y; e = 0.0
    for a, b in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
rows = []
for seed in range(5):
    Xs, ys = make_classification(20000, n_features=20, n_informative=8, n_classes=4,
                                 weights=[0.7, 0.2, 0.07, 0.03], flip_y=0.05, random_state=seed)
    Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.5, random_state=seed)
    Xcal, Xte, ycal, yte = train_test_split(Xb, yb, test_size=0.5, random_state=seed)
    un = LogisticRegression(max_iter=3000).fit(Xa, ya)
    bal = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xa, ya)
    zc, zt = bal.decision_function(Xcal), bal.decision_function(Xte)
    nll = lambda T: -np.log(softmax(zc / T)[np.arange(len(ycal)), ycal]).mean()
    T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
    prior = np.bincount(ya) / len(ya)
    pc = bal.predict_proba(Xte) * prior; pc /= pc.sum(1, keepdims=True)             # prior correction
    vs = LogisticRegression(max_iter=3000).fit(np.log(bal.predict_proba(Xcal)), ycal)  # vector scaling
    rows.append([ece(un.predict_proba(Xte), yte), ece(bal.predict_proba(Xte), yte), ece(softmax(zt / T), yte),
                 ece(pc, yte), ece(vs.predict_proba(np.log(bal.predict_proba(Xte))), yte),
                 bal.predict_proba(Xte)[:, 3].mean(), (yte == 3).mean(),
                 f1_score(yte, un.predict(Xte), average="macro"), f1_score(yte, bal.predict(Xte), average="macro")])
rows = np.array(rows)
names = ["ECE unweighted", "ECE balanced", "ECE +temperature", "ECE +prior corr.", "ECE +vector scaling",
         "mean p(rare) balanced", "rare class freq", "macro-F1 unweighted", "macro-F1 balanced"]
for n_, m_, s_ in zip(names, rows.mean(0), rows.std(0)):
    print(f"{n_:24s} {m_:.3f} ± {s_:.3f}")
