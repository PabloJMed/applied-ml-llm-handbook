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
] * 10
labels = ["restaurant", "restaurant", "restaurant", "pharmacy", "grocery", "restaurant",
          "restaurant", "restaurant", "restaurant", "pharmacy", "grocery", "pharmacy"] * 10
X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.3, random_state=0, stratify=labels)

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
