# 44 · Search ranking and learning to rank

Part XI · Applied ML systems (optional track). Read Chapter 43 first: this chapter reuses its funnel, two-tower retrieval and position-bias material. Worked design that builds on it: E.2 Dish and restaurant search. Retrieval for RAG (Chapter 30) uses the same lexical, semantic and hybrid methods.

## Why it matters

Search users have the strongest intent in the app: they typed what they want. Search usually converts far better than browsing, and a failed search (no results, wrong results) is one of the most direct ways to lose an order. In food delivery, search is also hard: short, misspelled, multilingual queries; results that must be open and deliverable right now; and a mix of restaurants and dishes on one page.

Learning to rank (LTR) is the core technique behind search ranking, and the same losses rank recommendations, ads and RAG passages. Interviewers expect you to know pointwise, pairwise and listwise approaches, explain LambdaMART, and design features for a marketplace search.

## Key concepts

- **Search pipeline**: query understanding (spelling, synonyms, intent) → retrieval (lexical and semantic) → ranking → business rules. [Eugene Yan: System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/).
- **Learning to rank (pointwise, pairwise, listwise)**: training a model to order results, with losses on single items, pairs or whole lists. [Learning to rank](https://en.wikipedia.org/wiki/Learning_to_rank).
- **LambdaMART**: boosted trees trained with pairwise gradients weighted by the change in NDCG; the standard strong baseline. [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/); available as the `lambdarank` objective in [LightGBM](https://lightgbm.readthedocs.io/en/latest/).
- **Relevance labels**: human judgements or behaviour (clicks, add-to-cart, orders) turned into graded labels.
- **Position bias and unbiased learning to rank**: top results get clicked partly because they are on top; correct with inverse propensity weighting. [Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468).
- **Offline ranking metrics and online tests**: NDCG and MRR offline, then interleaving or A/B tests online (Chapters 4 and 47).
- **Multilingual and transliterated queries**: spelling variants, mixed scripts and per-language vocabularies.
- **Textbook**: [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/).

## 44.1 Search vs recommendation, and the search pipeline

Search is a recommender with a query. The funnel from Chapter 43 still applies, with three differences:

- **Relevance to the query comes first.** A personalised result that does not match "sushi" is wrong, however much the user likes burgers. Personalisation breaks ties among relevant results.
- **Query understanding is a stage of its own.** Before retrieving anything, you need to know what the user meant.
- **Lexical matching matters.** Exact words (a restaurant's name, a dish name) are strong signals, so search keeps a keyword index next to the embedding index.

![Diagram](images/26-01.png)

*search pipeline · query understanding, hybrid retrieval, ranking, blending*

The two retrieval paths cover each other's weaknesses: BM25 finds "Pizzeria Napoli" exactly but misses "something cheesy"; embeddings find the vague query but can miss an exact brand name. The ranker then sees the merged set with features from both.

## 44.2 Query understanding

Food-delivery queries are short (one or two words), often misspelled, often in several languages, and ambiguous: "burger" is a dish, "Burger King" is a restaurant, "healthy" is a vague need. Query understanding turns the raw string into something retrieval and ranking can use.

| Step | What it does | Typical method |
|---|---|---|
| Normalisation | Lower-case, strip accents, unify units and spacing | Rules |
| Spelling correction | "piza" → "pizza", "suhsi" → "sushi" | Edit distance against a vocabulary weighted by popularity; learned from reformulations (users who typed A then B) |
| Intent classification | Restaurant name, dish, cuisine, grocery item, vague need | Text classifier (fine-tuned small encoder, or an LLM labelling data for one) |
| Entity extraction | "vegan", "under €10", "near me", a brand | Dictionaries plus NER; LLMs for long-tail phrasing |
| Query expansion | "chips" → "fries"; "poke" → "poke bowl" | Synonym lists mined from co-clicks; embeddings |

**Spelling correction from scratch** (the Norvig approach): generate all strings within one or two edits and pick the most popular known word.

```python
def edits1(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
    inserts = [a + c + b for a, b in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def correct(word, vocab_counts):
    """vocab_counts: word -> popularity (e.g. how often it appears in successful queries)."""
    if word in vocab_counts:
        return word
    one = [w for w in edits1(word) if w in vocab_counts]
    if one:
        return max(one, key=vocab_counts.get)
    two = {w2 for w1 in edits1(word) for w2 in edits1(w1) if w2 in vocab_counts}
    return max(two, key=vocab_counts.get) if two else word
vocab = {"pizza": 9000, "sushi": 7000, "burger": 8000, "lasagna": 900, "poke": 1200}
correct("pizzza", vocab), correct("lasgna", vocab), correct("suhsi", vocab)
# pizza, lasagna, sushi
```

Build the vocabulary from queries that led to orders, not from all queries, or common misspellings ("piza") end up in the vocabulary and are never corrected. In production, prefer showing results for the correction with a "search instead for" link over silently replacing the query.

**Autocomplete.** Suggest completions as the user types, ranked by popularity, and later by personalisation and location. The basic data structure is a prefix lookup with the top suggestions cached per prefix:

```python
import bisect

class Autocomplete:
    def __init__(self, query_counts):
        self.queries = sorted(query_counts)    # sorted strings make prefix ranges contiguous
        self.counts = query_counts

    def suggest(self, prefix, k=5):
        lo = bisect.bisect_left(self.queries, prefix)
        hi = bisect.bisect_left(self.queries, prefix + "\uffff")
        return sorted(self.queries[lo:hi], key=self.counts.get, reverse=True)[:k]

ac = Autocomplete({"pizza": 900, "pizza margherita": 300, "pizzeria napoli": 120, "poke": 500})
ac.suggest("piz")   # ['pizza', 'pizza margherita', 'pizzeria napoli']
```

At scale this is a trie or finite-state transducer with the top-k stored at each node, so a lookup costs one walk down the prefix. Rank suggestions with a small learned model (popularity in this city, this user's history, time of day), and filter offensive or zero-result completions.

## 44.3 Retrieval: lexical, semantic and hybrid

**BM25** is the standard keyword-scoring function. For each query term it multiplies an IDF weight (rare words count more) by a saturating term-frequency factor (the tenth mention adds less than the first), normalised by document length:

`BM25(q, d) = Σ over terms t in q of IDF(t) · f(t,d)·(k₁+1) / (f(t,d) + k₁·(1` `− b + b·|d|/avgdl))` with k₁ ≈ 1.2 (how fast term frequency saturates) and b ≈ 0.75 (how much length normalisation applies).

```python
import math
from collections import Counter

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
```

In production this is an inverted index in Elasticsearch, OpenSearch or Solr, with fields weighted differently (a match in the restaurant name counts more than one in a menu description).

**Semantic retrieval** embeds the query and the documents with a bi-encoder (the two-tower model of Chapter 43, with a text encoder for the query) and searches with ANN. Train it on (query, ordered item) pairs from search logs with in-batch negatives, starting from a pretrained multilingual sentence encoder. Chapter 29 covers embedding models in depth.

**Hybrid retrieval** runs both and fuses the lists. Scores from BM25 and cosine similarity are on different scales, so the simplest robust fusion uses only ranks: **reciprocal rank fusion (RRF)**.

```python
from collections import defaultdict

def rrf(rankings, k=60):
    """rankings: lists of doc ids, best first. k=60 is the value from the original paper."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

rrf([bm25_ids, embedding_ids])
```

RRF is a good first-stage merge. The ranker then sees both the BM25 score and the cosine similarity as features and learns the best weighting.

## 44.4 Features for search ranking

Search ranking features come in four families. In an interview, listing them by family shows structure.

| Family | Examples |
|---|---|
| Query | Length, intent class, language, is it a brand name, query popularity, historical click-through of the query |
| Document (restaurant or dish) | Rating, number of orders, price level, cuisine, prep time, cancellation rate, photo quality, is it new |
| Query-document | BM25 per field (name, dishes, tags), exact name match, cosine similarity, share of query terms matched, historical CTR and conversion for this query-document pair |
| User and context | User's past orders from this restaurant, cuisine affinity, price sensitivity, distance, ETA, time of day, device |

**Query-document engagement features are often the strongest** and also the riskiest. Raw CTR on few impressions is noise (2 clicks out of 3 impressions is not a 67% CTR), so smooth it towards a prior:

```python
prior = df["clicks"].sum() / df["impressions"].sum()
alpha = 50                                   # prior strength, in pseudo-impressions
qd = (df.groupby(["query", "restaurant_id"])
        .agg(clicks=("clicks", "sum"), imps=("impressions", "sum")))
qd["ctr_smoothed"] = (qd["clicks"] + alpha * prior) / (qd["imps"] + alpha)
```

Two cautions. Compute these features only from data before each training example's timestamp, or you leak the label into the feature. And engagement features carry position bias (results shown higher got more clicks), so either compute them from position-debiased clicks or accept that they reinforce the current ranking.

## 44.5 Learning to rank: pointwise, pairwise, listwise

All three train a scoring function s = f(query, document) and sort by s. They differ in what the loss looks at.

| Approach | Loss looks at | Idea | Examples | Trade-off |
|---|---|---|---|---|
| Pointwise | One document at a time | Predict its relevance label or click probability | Logistic regression, GBDT with logloss, MSE on grades | Simple, gives calibrated probabilities; ignores that only the order matters |
| Pairwise | Two documents for the same query | The more relevant one should score higher | RankNet, RankSVM, BPR | Optimises order directly; treats a swap at positions 50 and 51 like one at 1 and 2 |
| Listwise | The whole result list | Optimise a list metric (NDCG) or match a target distribution over the list | ListNet, LambdaRank, LambdaMART, softmax loss | Focuses on the top where users look; harder to reason about, scores not calibrated |

**Labels.** Graded relevance per (query, document): from human judgments (0 = irrelevant to 3 = perfect), or from engagement (0 = shown, 1 = clicked, 2 = added to cart, 3 = ordered). Ranking losses only compare documents **within the same query**, which is why the data carries a query or group id.

**The three losses in PyTorch,** for one query with scores `s` and graded labels `y`:

```python
import torch
import torch.nn.functional as F

def pointwise_loss(s, y):
    # Binary relevance: was it relevant at all?
    return F.binary_cross_entropy_with_logits(s, (y > 0).float())

def ranknet_loss(s, y):
    # For every pair where y_i > y_j: -log sigmoid(s_i - s_j)
    diff = s.unsqueeze(1) - s.unsqueeze(0)               # diff[i, j] = s_i - s_j
    pairs = (y.unsqueeze(1) > y.unsqueeze(0)).float()    # 1 where i should rank above j
    return (F.softplus(-diff) * pairs).sum() / pairs.sum().clamp(min=1)

def listnet_loss(s, y):
    # Cross-entropy between the label distribution and the score distribution over the list
    return -(F.softmax(y.float(), dim=0) * F.log_softmax(s, dim=0)).sum()
```

`softplus(-x)` is `log(1 + e^{-x})`, which equals `-log σ(x)` but is numerically stable. On a small synthetic test, all three learn a sensible ranking; the pairwise and listwise losses get the order right with fewer of the model's resources spent on getting the exact label values right.

## 44.6 From RankNet to LambdaMART

This is the most-asked LTR story in interviews. Tell it in three steps.

1. **RankNet (Burges et al., 2005).** Model the probability that document i should rank above j as σ(sᵢ − sⱼ) and train with cross-entropy on pairs. The gradient pushes sᵢ up and sⱼ down for each misordered pair; summing over pairs gives one number per document, called its **lambda**: how hard and in which direction to push its score.
2. **LambdaRank (2006).** NDCG is what we care about, but it is not differentiable (it depends on sorted positions). The trick: skip writing the loss and define the gradients directly as RankNet's pair gradient multiplied by **|ΔNDCG|**, the change in NDCG if i and j swapped positions. Swapping positions 1 and 2 changes NDCG a lot; swapping 50 and 51 barely at all, so the model concentrates on the top of the list.
3. **LambdaMART (2008–2010).** Use those lambdas as the pseudo-gradients in gradient-boosted trees (MART: multiple additive regression trees). Each new tree fits the lambdas. It won the 2010 Yahoo! Learning to Rank Challenge and is still the default production ranker for tabular features.

**LambdaRank in code:** the RankNet pair loss, weighted by |ΔNDCG| computed from the current ranking.

```python
def lambdarank_loss(s, y):
    n = len(s)
    with torch.no_grad():                                 # weights, not part of the gradient
        order = torch.argsort(s, descending=True)
        ranks = torch.empty(n, dtype=torch.long)
        ranks[order] = torch.arange(n)                    # current position of each doc
        gains = 2.0 ** y.float() - 1
        disc = 1.0 / torch.log2(ranks.float() + 2)
        ideal_disc = 1.0 / torch.log2(torch.arange(n).float() + 2)
        idcg = (torch.sort(gains, descending=True).values * ideal_disc).sum().clamp(min=1e-9)
        # |change in NDCG| if docs i and j swapped positions
        delta = ((gains.unsqueeze(1) - gains.unsqueeze(0)) *
                 (disc.unsqueeze(1) - disc.unsqueeze(0))).abs() / idcg
    pairs = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    diff = s.unsqueeze(1) - s.unsqueeze(0)
    return (delta * pairs * F.softplus(-diff)).sum() / pairs.sum().clamp(min=1)
```

**LambdaMART in practice** is `LGBMRanker(objective="lambdarank")` (code in Chapter 43.5) or XGBoost with `objective="rank:ndcg"`. Useful settings to know: `eval_at` (the k for NDCG), `label_gain` (the gain per relevance grade), and `lambdarank_truncation_level` (how deep in the list pairs are considered). Rows must be sorted by query, with `group` giving the number of rows per query.

## 44.7 Marketplace search: what is special in food delivery

- **Results must be orderable.** Open now, within delivery range, item in stock, minimum order reachable. A result the user cannot order is worse than no result. Apply these filters in retrieval (geo-filtered index per city or zone) rather than after ranking.
- **Navigational queries.** When someone types a restaurant name, that restaurant must be first. Detect it in query understanding (exact or fuzzy name match) and pin it. If it is closed, still show it, clearly labelled, with alternatives below: the user learns it exists and is not confused.
- **Restaurants vs dishes.** "Pizza" can mean restaurants that serve pizza or specific pizza dishes. Either rank both types in one model with the type as a feature, or rank them separately and blend using the intent probabilities from query understanding.
- **Distance, ETA and fees.** These are ranking features (users trade quality for speed) and user-facing sort options. Keep the default ranking learned; offer explicit sorts on top.
- **Zero and low results.** Track the zero-result rate by query. Fallbacks in order: spelling correction, query relaxation (drop a term), semantic retrieval, then related cuisines nearby. Most zero-result queries are long-tail misspellings, synonyms, or items nobody nearby sells, and each needs a different fix.
- **Multilingual queries and transliteration.** In the Gulf, users search in Arabic, English and transliterated Arabic ("shawarma", "shawerma", "شاورما"). Multilingual embeddings, transliteration variants in the synonym lists and per-language spelling vocabularies all help. It is a concrete point to raise in any interview for a multilingual market.
- **Sponsored results.** Ads are blended into fixed slots and should pass a relevance threshold so a paid burger does not appear for "sushi". Ad ranking is roughly bid × predicted conversion, with relevance as a constraint (Chapter 46, worked design E.7).
- **Supply health.** Pushing all "pizza" searchers to one popular place overloads its kitchen and lengthens delivery for everyone. Prep-time and load features, and occasional throttling of overloaded partners, protect the experience.

## 44.8 Neural re-rankers: cross-encoders and LLMs

**Bi-encoder vs cross-encoder.** A bi-encoder embeds query and document separately and compares vectors: fast, precomputable, used for retrieval. A cross-encoder feeds the query and document together into one transformer, so every query token can attend to every document token. It is much more accurate at judging relevance, but it must run once per (query, document) pair, so it is only affordable on the top 50–100 candidates.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # small English model; pick a multilingual one for Arabic/English
scores = reranker.predict([(query, doc_text) for doc_text in top_candidates])
```

(API as of 2025–26; check the current sentence-transformers docs.) For marketplace search, the cross-encoder's score usually becomes **a feature in the LambdaMART ranker** rather than the final ranking, because availability, distance and popularity still matter. Fine-tune it on your own (query, ordered item) pairs; off-the-shelf models are trained on web search data.

**ColBERT (late interaction)** sits between the two: it stores one vector per document token and scores with the sum of maximum similarities per query token. Nearly cross-encoder quality at retrieval speed, at the cost of a much larger index. Chapter 30 covers it.

**LLMs in search ranking.**

- *Pointwise:* ask the LLM to grade each (query, document) pair. Simple, but expensive per document.
- *Listwise:* give the LLM the query and a short numbered list and ask for the reordered list (the RankGPT approach). Sliding windows handle longer lists.
- *As a label source:* the most practical use today. LLM-graded relevance labels for thousands of queries train or evaluate the production ranker, after calibration against human judgments (Chapter 24, worked design E.6).
- *Distillation:* train a small cross-encoder to imitate the LLM's judgments, then serve the small model.

Cost and latency usually keep LLMs out of the online path for high-traffic search, except for long, natural-language queries where the gain is largest.

## 44.9 Evaluating search

**Where relevance labels come from.**

| Source | Strength | Weakness |
|---|---|---|
| Clicks and orders | Free, abundant, reflect real users | Position bias; only covers what was shown; an order also depends on price and ETA, not only relevance |
| Human judgments | Clean graded labels with written guidelines | Expensive; slow; raters need clear guidelines and agreement checks |
| LLM judgments | Cheap at scale; consistent | Must be calibrated against human labels (Cohen's kappa) and checked for bias |

A practical setup: a fixed **golden set** of queries (the head queries plus a random sample of torso and tail), judged by humans on a 0–3 scale; LLM judgments to extend it; engagement data for training.

**Offline metrics.**

- **NDCG@k with graded gains** (2^rel − 1): the default for ranking quality.
- **MRR**: for navigational queries, where one result is the right one (the restaurant the user named).
- **Precision@k, Recall@k**: precision for the visible top; recall for the retrieval stage.

```python
import numpy as np

def dcg_at_k(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))

def ndcg_at_k(rels_in_ranked_order, k, all_rels=None):
    """all_rels: grades of ALL known relevant docs for the query, so missed docs lower the score."""
    pool = all_rels if all_rels is not None else rels_in_ranked_order
    ideal = dcg_at_k(sorted(pool, reverse=True), k)
    return dcg_at_k(rels_in_ranked_order, k) / ideal if ideal > 0 else 0.0

def mrr(first_relevant_ranks):
    """1-based rank of the first relevant result per query, or None if there was none."""
    return float(np.mean([1.0 / r if r else 0.0 for r in first_relevant_ranks]))

ndcg_at_k([0, 3, 2, 0, 1], k=5)   # ~0.67
mrr([1, 3, None, 2])              # ~0.46
```

A common bug: computing the ideal DCG only from the retrieved list. Then a system that retrieves one mediocre result and nothing else scores a perfect 1.0. Use all known relevant documents for the ideal.

**Click models** explain how users produce clicks, so you can recover relevance from biased logs. The cascade model assumes users scan from the top and click the first relevant result; the dynamic Bayesian network (DBN) model adds the chance of continuing after a click. They are also a way to estimate the position propensities used in Chapter 43.9.

**Online metrics.**

| Metric | What it tells you |
|---|---|
| Search conversion | Share of searches that end in an order: the primary metric |
| Zero-result rate | Share of queries with no results |
| Reformulation rate | Share of searches followed by another search: a sign the first one failed |
| Abandonment | Searches with no click at all |
| Position of first click or order | Lower is better: relevant results are near the top |
| Time to order from search | Speed of a successful search |

For comparing two rankers, **team-draft interleaving** is the standard tool in search: build each result list by alternately drafting from the two rankers, and credit each click or order to the ranker that contributed that result. It detects differences with much less traffic than an A/B test. Use A/B tests for the final decision on business metrics, and always segment by query type (head, torso, tail; navigational vs dish vs cuisine), because averages hide that a change helped head queries and broke the tail.

## Pitfalls

- **Ideal DCG from the retrieved list only.** It rewards retrieving less. Use all known relevant documents.
- **Mixing queries in the ranking loss.** Pairwise and listwise losses only make sense within a query; forgetting `group` or sorting rows wrongly silently breaks training.
- **Leaky engagement features.** Query-document CTR computed over a period that includes the training example's own click.
- **Treating clicks as relevance.** Clicks carry position bias, and orders also depend on price, ETA and fees. Debias, or combine with human or LLM judgments.
- **Silent spelling correction.** Replacing a correct but rare word (a new restaurant's name) with a popular one. Show "showing results for", and never correct exact matches to known names.
- **Only evaluating on head queries.** Head queries are easy and already good; the tail is where users fail. Sample the tail into your golden set.
- **Semantic retrieval alone.** Embeddings miss exact names and numbers. Keep a lexical path.
- **Ignoring navigational intent.** If the named restaurant is not first, nothing else in the list matters to that user.
- **Results that cannot be ordered.** Filtering after ranking leaves empty or closed pages. Filter in retrieval.
- **One metric for all intents.** NDCG for dish and cuisine queries, MRR or success@1 for navigational ones.

## Interview questions

Questions 1–5 are basic, 6–11 intermediate, 12–15 senior.

**1. What is the difference between pointwise, pairwise and listwise learning to rank?** Pointwise predicts each document's relevance on its own (classification or regression). Pairwise learns which of two documents for the same query should rank higher (RankNet). Listwise optimises a property of the whole list, usually NDCG (LambdaRank, LambdaMART, ListNet). Pairwise and listwise optimise order, which is what users see; pointwise gives calibrated probabilities.

**2. Explain BM25 in one minute.** A keyword score summing, over query terms, IDF (rare terms count more) times a term-frequency factor that saturates (k₁) and is normalised by document length (b). It is a strong, cheap baseline and still the lexical half of most hybrid systems.

**3. Why combine lexical and semantic retrieval?** Lexical catches exact names, brands and numbers; semantic catches synonyms, paraphrases and vague queries. Each covers the other's failures. Fuse with reciprocal rank fusion, then let the ranker weigh both scores as features.

**4. What is NDCG and why the log discount?** DCG sums graded gains discounted by log₂(position + 1), because users look less at lower positions; NDCG divides by the best possible DCG so scores are comparable across queries. Graded gains (2^rel − 1) reward putting the very best results first.

**5. How would you handle misspelled queries?** Candidate corrections within one or two edits, ranked by popularity in successful queries and by context; learn corrections from reformulations (users who typed A then quickly B and ordered). Show the correction transparently and never overwrite exact matches to known restaurant names.

**6. Explain LambdaMART.** RankNet gives pairwise gradients from σ(sᵢ − sⱼ). LambdaRank multiplies each pair's gradient by the |ΔNDCG| of swapping the pair, so errors near the top matter most, without needing NDCG to be differentiable. LambdaMART fits gradient-boosted trees to those lambdas. It is the default production ranker for tabular features.

**7. Bi-encoder vs cross-encoder: when do you use each?** Bi-encoders encode query and documents separately, so document vectors are precomputed and searched with ANN: retrieval. Cross-encoders read query and document together: far more accurate but one forward pass per pair, so only for re-ranking the top 50–100. In a marketplace, the cross-encoder score is often a feature for the final ranker.

**8. Which features would you build for restaurant search?** By family: query (intent, length, language), document (rating, popularity, prep time, price), query-document (BM25 per field, exact name match, embedding similarity, smoothed query-restaurant CTR and conversion), user and context (distance, ETA, past orders there, cuisine affinity, time of day).

**9. A user types a restaurant name and it is closed. What do you show?** The restaurant first, clearly marked closed with its opening time, then the most similar open alternatives nearby. Hiding it makes the user think search is broken.

**10. How do you get relevance labels for training and evaluation?** Engagement for training at scale, debiased for position; a human-judged golden set (0–3 scale, guidelines, agreement checks) for evaluation; LLM judges to scale the golden set after calibrating against the human labels.

**11. How do you measure search quality online?** Search conversion as the primary metric, with zero-result rate, reformulation rate, abandonment and position of first order as diagnostics; team-draft interleaving to compare rankers quickly; A/B tests for the final call; everything segmented by query type.

**12. Search conversion dropped 2% last week. How do you investigate?** Check instrumentation and traffic mix first (app release, new city, marketing campaign bringing new users). Then segment: by query type, language, city, platform. Look at zero-result and reformulation rates for spikes, which point to retrieval or spelling breaking. Check supply (restaurants closed, a big partner offline) and recent ranking or index changes. Compare with conversion from browsing to separate a search problem from a demand problem.

**13. How would you blend restaurants and dishes in one result page?** Estimate intent probabilities in query understanding. Either rank both types with one model that includes type and intent features, calibrated so scores are comparable, or rank separately and allocate slots by intent (more dishes for "margherita", more restaurants for "italian"). Test the blending policy online; offline metrics struggle with mixed lists.

**14. How would you add LLMs to restaurant search?** Start offline: LLM-generated labels to train and evaluate the ranker, menu enrichment for better matching, synonym and transliteration mining. Online, use them for query understanding of long natural-language queries ("something light and vegan under €12") turned into filters plus a semantic query. Keep the high-traffic ranking path on LambdaMART and distilled cross-encoders for latency and cost.

**15. Your new ranker wins offline on NDCG but loses an interleaving test. Why might that be?** The offline labels and online behaviour disagree: the golden set over-represents head queries; labels judge topical relevance while users also weigh ETA, price and fees; the offline candidate set ignores availability; or the new model is worse at navigational queries, where one failure is very visible. Look at per-segment interleaving results and at the queries where the two rankers disagree most.

## Practice

1. **BM25 from scratch.** Implement BM25 without looking (30 minutes). Test it on a small

set of menus you write yourself, then compare with the `rank_bm25` package.

2. **Learning to rank on a public dataset.** Download MSLR-WEB10K (Microsoft's LTR dataset, pre-split into folds with query groups). Train LightGBM with `objective="binary"` (pointwise) and `objective="lambdarank"` and compare NDCG@10. *Good solution:* a short table, and an explanation of why lambdarank wins at the top.
3. **Losses by hand.** Implement the RankNet, ListNet and LambdaRank losses from 44.5–44.6 and train a linear scorer with each on MSLR features. Plot NDCG@10 per epoch.
4. **Hybrid retrieval.** On the Amazon ESCI shopping-queries dataset (query-product pairs graded Exact, Substitute, Complement, Irrelevant), retrieve with BM25 and with a sentence-transformers model, fuse with RRF, and compare Recall@100 and NDCG@10 for each. *Good solution:* hybrid beats both; you can show queries where each single method fails.
5. **Re-rank with a cross-encoder** on the top 50 from exercise 4 and measure the NDCG gain and the extra latency per query.
6. **Spelling and autocomplete.** Build the spelling corrector and autocomplete from 44.2 on a list of dish names, and write five test cases each, including one that must not be corrected.
7. **Design rehearsal.** Answer "design dish and restaurant search for a delivery app" aloud in 35 minutes, then compare with worked design E.2.

## Read next

- Manning, Raghavan and Schütze, *Introduction to Information Retrieval* (free online): chapter 6 (scoring and term weighting), chapter 8 (evaluation), chapter 15 (machine-learned relevance).
- Burges, *From RankNet to LambdaRank to LambdaMART: An Overview* (Microsoft Research technical report, 2010). The single best source for 44.6.
- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- Cormack, Clarke and Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and* *individual Rank Learning Methods* (SIGIR 2009).
- Nogueira and Cho, *Passage Re-ranking with BERT* (2019): the cross-encoder re-ranker.
- Sun et al., *Is ChatGPT Good at Search? Investigating Large Language Models as Re-* *Ranking Agents* (RankGPT, 2023).
- Chuklin, Markov and de Rijke, *Click Models for Web Search* (2015 book); Chapelle and Zhang, *A Dynamic Bayesian Network Click Model for Web Search Ranking* (2009).
- Radlinski, Kurup and Joachims, *How Does Clickthrough Data Reflect Retrieval* *Quality?* (2008), for interleaving.
- Eugene Yan's writing on search and query understanding (eugeneyan.com), and the engineering blogs of Instacart, DoorDash and Uber Eats on search ranking, which describe very similar marketplace problems.

**Links:** [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) · [Unbiased Learning-to-Rank](https://arxiv.org/abs/1608.04468) · [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/). Other titles above can be found by searching their title.

## Done when

You can draw the search pipeline and explain what query understanding adds.

You can write BM25 and RRF from scratch and explain k₁ and b. You can write pointwise, RankNet and ListNet losses and say when each fits. You can tell the RankNet → LambdaRank → LambdaMART story in two minutes. You can list search features by family for a restaurant marketplace. You can explain bi-encoder vs cross-encoder and where an LLM fits in search. You can compute NDCG and MRR correctly, including the ideal-DCG trap. You can name five online search metrics and explain team-draft interleaving.

You can debug a 2% drop in search conversion out loud, step by step.

---

[← 43 · Recommender systems](43-recommender-systems.md) · [Contents](../README.md#contents) · [45 · Marketplace and logistics ML →](45-marketplace-and-logistics-ml.md)
