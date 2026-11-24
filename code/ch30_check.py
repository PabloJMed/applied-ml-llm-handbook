import math, time
from collections import Counter, defaultdict
import numpy as np
import torch

# ---- inverted index with TF-IDF
class InvertedIndex:
    def __init__(self, docs):
        self.postings = defaultdict(list)          # term -> [(doc_id, term_count)]
        self.n_docs = len(docs)
        for doc_id, text in enumerate(docs):
            for term, count in Counter(text.lower().split()).items():
                self.postings[term].append((doc_id, count))

    def idf(self, term):
        df = len(self.postings.get(term, []))
        return math.log((1 + self.n_docs) / (1 + df)) + 1    # smoothed, as in scikit-learn

    def search(self, query, k=10):
        scores = defaultdict(float)
        for term in query.lower().split():
            w = self.idf(term)
            for doc_id, count in self.postings.get(term, []):   # only docs containing the term
                scores[doc_id] += (1 + math.log(count)) * w       # sublinear tf x idf
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]

docs = ["refund policy for late orders", "how to change delivery address",
        "late delivery compensation voucher", "allergy information for dishes"]
idx = InvertedIndex(docs)
print(idx.search("late delivery"))

# ---- exact search + IVF from scratch
rng = np.random.default_rng(0)
d, n, nq = 64, 20000, 200
centers = rng.normal(size=(50, d))
X = (centers[rng.integers(0, 50, n)] + 0.5 * rng.normal(size=(n, d))).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = X[rng.choice(n, nq, replace=False)] + 0.05 * rng.normal(size=(nq, d)).astype("float32")
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

def exact_topk(Q, X, k=10):
    S = Q @ X.T
    return np.argsort(-S, axis=1)[:, :k]

def kmeans(X, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            pts = X[assign == j]
            if len(pts):
                C[j] = pts.mean(0)
    return C, assign

class IVFIndex:
    def __init__(self, X, n_lists=100, seed=0):
        sample = X[np.random.default_rng(seed).choice(len(X), min(len(X), 5000), replace=False)]
        C, _ = kmeans(sample, n_lists, iters=10, seed=seed)
        self.centroids = C / np.linalg.norm(C, axis=1, keepdims=True)   # unit centroids: inner product = cosine
        assign = np.argmax(X @ self.centroids.T, axis=1)
        self.lists = [np.flatnonzero(assign == j) for j in range(n_lists)]
        self.X = X

    def search(self, q, k=10, n_probe=8):
        nearest_lists = np.argsort(-(self.centroids @ q))[:n_probe]
        cand = np.concatenate([self.lists[j] for j in nearest_lists])
        scores = self.X[cand] @ q
        return cand[np.argsort(-scores)[:k]]

def recall_vs_exact(approx, exact):
    return np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)])

truth = exact_topk(Q, X)
ivf = IVFIndex(X, n_lists=100)
for n_probe in [1, 4, 16]:
    approx = [ivf.search(q, n_probe=n_probe) for q in Q]
    print("IVF nprobe", n_probe, "recall@10", round(recall_vs_exact(approx, truth), 3))

# ---- product quantisation from scratch
class PQ:
    def __init__(self, X, m=8, k=256, seed=0):
        self.m, self.ds = m, X.shape[1] // m
        self.codebooks = []
        codes = np.empty((len(X), m), dtype=np.uint8)
        for i in range(m):
            sub = X[:, i * self.ds:(i + 1) * self.ds]
            sample = sub[np.random.default_rng(seed).choice(len(sub), 4000, replace=False)]
            C, _ = kmeans(sample, k, iters=8, seed=seed)
            self.codebooks.append(C)
            codes[:, i] = np.argmin(((sub[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        self.codes = codes

    def search(self, q, k=10):
        # asymmetric distance: precompute q-subvector to every centroid, then look up
        tables = np.stack([self.codebooks[i] @ q[i * self.ds:(i + 1) * self.ds] for i in range(self.m)])  # (m, k)
        scores = tables[np.arange(self.m), self.codes].sum(axis=1)     # approximate dot product per vector
        return np.argsort(-scores)[:k]

t = time.time()
pq = PQ(X, m=16, k=256)
print("pq build", round(time.time() - t, 1), "s; bytes per vector", pq.codes.shape[1], "vs", X.shape[1] * 4)
approx = [pq.search(q) for q in Q]
print("PQ recall@10", round(recall_vs_exact(approx, truth), 3))
def pq_search_rerank(pq, X, q, k=10, n_candidates=100):
    cand = pq.search(q, k=n_candidates)
    return cand[np.argsort(-(X[cand] @ q))[:k]]
approx = [pq_search_rerank(pq, X, q) for q in Q]
print("PQ+rerank recall@10", round(recall_vs_exact(approx, truth), 3))

# ---- FAISS
import faiss
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.add(X)
index.hnsw.efSearch = 64
_, I = index.search(Q, 10)
print("HNSW recall@10", round(recall_vs_exact(I, truth), 3))
for ef in (8, 16, 32, 64, 128):                       # recall against efSearch
    index.hnsw.efSearch = ef
    _, I = index.search(Q, 10)
    print("HNSW efSearch", ef, "recall@10", round(recall_vs_exact(I, truth), 3))
index.hnsw.efSearch = 64
# why PQ alone gets ~0.33: its score error compared with the gaps between neighbours' scores
S = Q @ X.T
approx_scores = np.stack([np.stack([pq.codebooks[i] @ q[i * pq.ds:(i + 1) * pq.ds] for i in range(pq.m)])[np.arange(pq.m), pq.codes].sum(1) for q in Q])
srt = -np.sort(-S, axis=1)
print(f"PQ score error std {np.std(approx_scores - S):.3f}; median gap 10th-11th {np.median(srt[:, 9] - srt[:, 10]):.4f}; "
      f"mean score 10th {srt[:, 9].mean():.3f}, 100th {srt[:, 99].mean():.3f}")

quantizer = faiss.IndexFlatIP(d)
ivfpq = faiss.IndexIVFPQ(quantizer, d, 100, 16, 8, faiss.METRIC_INNER_PRODUCT)
ivfpq.train(X)
ivfpq.add(X)
ivfpq.nprobe = 16
_, I = ivfpq.search(Q, 10)
print("IVFPQ recall@10", round(recall_vs_exact(I, truth), 3))

# ---- ColBERT MaxSim
def maxsim(q_tok, d_tok):
    """q_tok: (Lq, dim), d_tok: (Ld, dim), both L2-normalised."""
    return (q_tok @ d_tok.T).max(dim=1).values.sum()
qt = torch.nn.functional.normalize(torch.randn(5, 128), dim=-1)
dt = torch.nn.functional.normalize(torch.randn(40, 128), dim=-1)
print("maxsim", maxsim(qt, dt).item())

# ---- retrieval metrics
def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)
def mrr_at_k(retrieved, relevant, k):
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0
print(recall_at_k([3, 1, 7], {1, 9}, 3), mrr_at_k([3, 1, 7], {1, 9}, 3))

# ---- filtered search: post-filter oversampling
city = rng.choice(["BCN", "MAD", "DXB", "LIS"], n)
def post_filter_search(index, q, allowed_mask, k=10, oversample=4):
    _, I = index.search(q[None], k * oversample)
    hits = [i for i in I[0] if i >= 0 and allowed_mask[i]]
    return hits[:k]
mask = city == "DXB"
res = post_filter_search(index, Q[0], mask)
print("post-filter got", len(res))
short = np.mean([len(post_filter_search(index, q, mask)) < 10 for q in Q])
from scipy.stats import binom
print(f"share of queries short of 10: {short:.2f}; binomial P(X < 10 | n=40, p=0.25) = {binom.cdf(9, 40, 0.25):.2f}")
for s in (0.5, 0.1, 0.01):
    o = next(o for o in range(1, 2000) if binom.sf(9, 10 * o, s) >= 0.95)
    print(f"selectivity {s}: oversample x{o} for 95% of queries to get 10")
# FAISS pre-filter with IDSelector (supported by flat/IVF/HNSW in recent versions)
sel = faiss.IDSelectorBatch(np.flatnonzero(mask).astype("int64"))
params = faiss.SearchParametersHNSW(sel=sel, efSearch=64)
_, I = index.search(Q[:1], 10, params=params)
print("pre-filter", all(mask[i] for i in I[0]))

# 30.11 sizing an index for 50M chunks of 768 dimensions (HNSW M=32: about 2M neighbour ids of 4 bytes per vector at layer 0)
N, d = 50e6, 768
links = N * 2 * 32 * 4 / 1e9
for name, bpv in (("float32", d * 4), ("int8", d), ("Matryoshka 256 dims, int8", 256)):
    print(f"HNSW over {name}: vectors {N * bpv / 1e9:.1f} GB + links {links:.1f} GB = {N * bpv / 1e9 + links:.1f} GB")
print(f"IVF-PQ at 64 bytes: {N * 64 / 1e9:.1f} GB; exact search: {N * d / 1e9:.1f}e9 multiply-adds per query")
print(f"re-scoring 200 candidates from float32 on SSD: {200 * d * 4 / 1e6:.2f} MB read per query")
print(f"embedding 50M chunks at 20k chunks/min per GPU: {N / 20000 / 60:.0f} GPU-hours")

# exact search is memory-bandwidth bound: one query reads every vector
print(f"bytes read per exact query: float32 {N * d * 4 / 1e9:.1f} GB -> {N * d * 4 / 100e9:.2f} s at 100 GB/s; int8 {N * d / 100e9:.2f} s")
print(f"GPU brute force: fp16 corpus {N * d * 2 / 1e9:.1f} GB, one pass at 3.35 TB/s = {N * d * 2 / 3.35e12 * 1e3:.0f} ms; "
      f"200 QPS needs {200 * 2 * N * d / 1e12:.1f} TFLOP/s")

# 30.4 HNSW level arithmetic: P(level >= l) = M^-l, so about log_M N layers
M = 32
print(f"layers for 50M at M=32: {math.log(N) / math.log(M):.2f}; share above layer 0: {1 / M:.4f}; "
      f"expected extra upper-layer links per node: {M / (M - 1):.2f}; bytes per vector (768-d float32, 2M links) {768 * 4 + 2 * M * 4}")

# 30.2 BM25 IDF and a worked score: 3 documents, query "late refund", k1 = 1.2, b = 0.75
docs3 = [s.split() for s in ["late order refund", "refund policy refund rules for every order", "delivery address change"]]
avgdl = sum(map(len, docs3)) / len(docs3)
def idf_bm25(term):
    df = sum(term in doc for doc in docs3)
    return math.log(1 + (len(docs3) - df + 0.5) / (df + 0.5))
def bm25(query, doc, k1=1.2, b=0.75):
    s = 0.0
    for term in query.split():
        tf = doc.count(term)
        if tf:
            s += idf_bm25(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
    return s
print("avgdl", round(avgdl, 3), "IDF late/refund/order", [round(idf_bm25(w), 3) for w in ("late", "refund", "order")],
      "classic IDF at df=2:", round(math.log((3 - 2 + 0.5) / (2 + 0.5)), 3))
print("BM25 scores", [round(bm25("late refund", doc), 3) for doc in docs3])

# 30.9 nDCG@5 with graded gains 2^rel - 1
def dcg(rels): return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
rels = [3, 0, 2, 1, 0]
print(f"DCG {dcg(rels):.3f}, ideal {dcg(sorted(rels, reverse=True)):.3f}, nDCG@5 {dcg(rels) / dcg(sorted(rels, reverse=True)):.3f}")
print(f"growth 3%/month for 12 months: x{1.03 ** 12:.2f}, int8 HNSW {51.2 * 1.03 ** 12:.0f} GB")
