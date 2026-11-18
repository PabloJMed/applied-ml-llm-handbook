import math, bisect
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [d.lower().split() for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query, i):
        tf, dl = self.tf[i], len(self.docs[i])
        s = 0.0
        for t in query.lower().split():
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, k=10):
        scored = sorted(((self.score(query, i), i) for i in range(len(self.docs))), reverse=True)
        return [i for s, i in scored[:k] if s > 0]

docs = ["Pizzeria Napoli margherita pizza calzone",
        "Sushi Bar Tokyo sushi ramen gyoza",
        "Burger House cheeseburger fries",
        "Napoli Express pizza pasta lasagna",
        "Green Bowl vegan poke salad"]
bm = BM25(docs)
print(bm.search("pizza napoli"))

def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
print(rrf([[0, 3, 2], [3, 4, 0]]))

def edits1(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
    inserts = [a + c + b for a, b in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def correct(word, vocab_counts):
    if word in vocab_counts:
        return word
    one = [w for w in edits1(word) if w in vocab_counts]
    if one:
        return max(one, key=vocab_counts.get)
    two = {w2 for w1 in edits1(word) for w2 in edits1(w1) if w2 in vocab_counts}
    return max(two, key=vocab_counts.get) if two else word

vocab = {"pizza": 9000, "piza": 3, "sushi": 7000, "burger": 8000, "lasagna": 900, "poke": 1200}
print(correct("piza", vocab), correct("pizzza", vocab), correct("lasgna", vocab), correct("suhsi", vocab))

class Autocomplete:
    def __init__(self, query_counts):
        self.queries = sorted(query_counts)
        self.counts = query_counts
    def suggest(self, prefix, k=5):
        lo = bisect.bisect_left(self.queries, prefix)
        hi = bisect.bisect_left(self.queries, prefix + "￿")
        return sorted(self.queries[lo:hi], key=self.counts.get, reverse=True)[:k]
ac = Autocomplete({"pizza": 900, "pizza margherita": 300, "pizzeria napoli": 120, "poke": 500, "pad thai": 200})
print(ac.suggest("piz"), ac.suggest("p", 3))

df = pd.DataFrame({"query": ["pizza", "pizza", "pizza", "sushi"], "restaurant_id": [1, 1, 2, 3],
                   "impressions": [100, 50, 3, 80], "clicks": [20, 5, 2, 10]})
prior = df["clicks"].sum() / df["impressions"].sum()
alpha = 50
qd = (df.groupby(["query", "restaurant_id"])
        .agg(clicks=("clicks", "sum"), imps=("impressions", "sum")))
qd["ctr_smoothed"] = (qd["clicks"] + alpha * prior) / (qd["imps"] + alpha)
print(qd)

def pointwise_loss(s, y):
    return F.binary_cross_entropy_with_logits(s, (y > 0).float())

def ranknet_loss(s, y):
    diff = s.unsqueeze(1) - s.unsqueeze(0)
    pairs = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    return (F.softplus(-diff) * pairs).sum() / pairs.sum().clamp(min=1)

def listnet_loss(s, y):
    return -(F.softmax(y.float(), dim=0) * F.log_softmax(s, dim=0)).sum()

def lambdarank_loss(s, y):
    n = len(s)
    with torch.no_grad():
        order = torch.argsort(s, descending=True)
        ranks = torch.empty(n, dtype=torch.long)
        ranks[order] = torch.arange(n)
        gains = 2.0 ** y.float() - 1
        disc = 1.0 / torch.log2(ranks.float() + 2)
        ideal_disc = 1.0 / torch.log2(torch.arange(n).float() + 2)
        idcg = (torch.sort(gains, descending=True).values * ideal_disc).sum().clamp(min=1e-9)
        delta = ((gains.unsqueeze(1) - gains.unsqueeze(0)) *
                 (disc.unsqueeze(1) - disc.unsqueeze(0))).abs() / idcg
    pairs = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    diff = s.unsqueeze(1) - s.unsqueeze(0)
    return (delta * pairs * F.softplus(-diff)).sum() / pairs.sum().clamp(min=1)

torch.manual_seed(0)
X = torch.randn(30, 8); w_true = torch.randn(8)
y = torch.bucketize(X @ w_true, torch.tensor([-0.5, 0.5, 1.5]))
for fn in [pointwise_loss, ranknet_loss, listnet_loss, lambdarank_loss]:
    w = torch.zeros(8, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    for _ in range(200):
        loss = fn(X @ w, y); opt.zero_grad(); loss.backward(); opt.step()
    print(fn.__name__, round(loss.item(), 4), torch.corrcoef(torch.stack([X @ w.detach(), X @ w_true]))[0, 1].item())

def dcg_at_k(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))

def ndcg_at_k(rels_in_ranked_order, k, all_rels=None):
    ideal = dcg_at_k(sorted(all_rels if all_rels is not None else rels_in_ranked_order, reverse=True), k)
    return dcg_at_k(rels_in_ranked_order, k) / ideal if ideal > 0 else 0.0

def mrr(first_relevant_ranks):
    return float(np.mean([1.0 / r if r else 0.0 for r in first_relevant_ranks]))

print(ndcg_at_k([0, 3, 2, 0, 1], k=5), mrr([1, 3, None, 2]))
