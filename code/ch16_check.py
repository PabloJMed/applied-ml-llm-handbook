"""Chapter 16 · Pretraining data: runnable checks (NumPy + scikit-learn, seconds)."""
import re
import hashlib
import numpy as np
from collections import Counter

rng = np.random.default_rng(0)

docs = {
    "article": "The committee published its report on Tuesday. It found that rainfall in the region had "
               "fallen by a fifth over two decades, and recommended new reservoirs and stricter limits on "
               "groundwater use. Farmers' groups welcomed the plan but asked for help with the costs.",
    "boilerplate": "Home | About | Contact | Login | Cart (0) | Home | About | Contact | Login | Cart (0) | "
                   "Copyright 2024 All rights reserved | Privacy | Terms | Cookies | Home | About",
    "spam": "best price best price best price buy now buy now cheap cheap cheap click here click here "
            "best price buy now cheap click here best price buy now",
    "near_dup": "The committee published its report on Tuesday. It found that rainfall in the region had "
                "fallen by a fifth over two decades, and recommended new reservoirs and stricter limits on "
                "groundwater use. Farmers' groups welcomed the plan but asked for support with the costs.",
    "code": "def mean(xs):\n    return sum(xs) / len(xs)\n\nprint(mean([1, 2, 3]))",
}

# 16.3 heuristic filters in the spirit of the Gopher and C4 rules
STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}
def heuristics(text):
    words = re.findall(r"\w+", text.lower())
    ngrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    dup3 = 1 - len(set(ngrams)) / max(1, len(ngrams))
    return {
        "words": len(words),
        "mean_word_len": round(float(np.mean([len(w) for w in words])), 2) if words else 0,
        "symbol_ratio": round(sum(c in "|#{}<>" for c in text) / max(1, len(words)), 2),
        "stopwords": sum(w in STOP for w in words),
        "dup_3gram_frac": round(dup3, 2),
    }
def keep(h):
    return h["words"] >= 30 and 3 <= h["mean_word_len"] <= 10 and h["symbol_ratio"] < 0.1 \
        and h["stopwords"] >= 2 and h["dup_3gram_frac"] < 0.3
for name, text in docs.items():
    h = heuristics(text)
    print(f"{name:12s} keep={keep(h)!s:5s} {h}")

# 16.5 MinHash near-duplicate detection
def shingles(text, k=5):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}
def minhash(sh, n=128):
    sig = []
    for seed in range(n):
        sig.append(min(int(hashlib.blake2b(f"{seed}:{s}".encode(), digest_size=8).hexdigest(), 16) for s in sh))
    return np.array(sig)
a, b, c = shingles(docs["article"]), shingles(docs["near_dup"]), shingles(docs["spam"])
true_j = len(a & b) / len(a | b)
sa, sb, sc = minhash(a), minhash(b), minhash(c)
print(f"Jaccard(article, near_dup): true {true_j:.3f}, MinHash estimate {(sa == sb).mean():.3f}; "
      f"article vs spam estimate {(sa == sc).mean():.3f}")
# LSH banding: b bands of r rows; a pair becomes a candidate if any band matches exactly
def p_candidate(s, bands, rows):
    return 1 - (1 - s ** rows) ** bands
for s in (0.5, 0.7, 0.8, 0.9):
    print(f"similarity {s}: P(candidate) with 16 bands x 8 rows = {p_candidate(s, 16, 8):.3f}; "
          f"20 x 5 = {p_candidate(s, 20, 5):.3f}")
bands = [tuple(sa[i * 8:(i + 1) * 8]) == tuple(sb[i * 8:(i + 1) * 8]) for i in range(16)]
print("article/near_dup: bands matching", sum(bands), "of 16 ->", "candidate" if any(bands) else "not candidate")

# 16.4 a quality classifier (fastText-style: hashed n-grams + linear model)
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
good = ["The study measured the effect of temperature on enzyme activity across five trials.",
        "In 1848 revolutions spread across Europe, driven by economic hardship and demands for reform.",
        "To compute the derivative, apply the chain rule to the outer and inner functions in turn.",
        "The algorithm sorts the list by repeatedly merging sorted halves until one list remains.",
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
        "The court ruled that the contract was void because one party had been misled."]
bad = ["CLICK HERE best deals!!! free free free subscribe now", "lol ok same tbh idk whatever",
       "Home About Contact Login Cart Privacy Terms", "buy cheap pills online no prescription best price",
       "you won't BELIEVE what happened next number 7 will shock you", "asdf asdf qwerty test test test"]
vec = HashingVectorizer(ngram_range=(1, 2), n_features=2**16, alternate_sign=False, norm="l2")
clf = LogisticRegression(C=10, max_iter=1000).fit(vec.transform(good + bad), [1] * 6 + [0] * 6)
for t in ["The report describes how rainfall and groundwater use changed over two decades.",
          "free deals click here subscribe now best price"]:
    print(f"quality score {clf.predict_proba(vec.transform([t]))[0, 1]:.2f}: {t[:55]}")

# 16.6 decontamination by 13-gram overlap
def ngrams(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
bench = ("A train leaves the station at 3 pm travelling at 60 miles per hour towards a city "
         "that is 150 miles away. At what time does it arrive?")
web = ("Homework help: a train leaves the station at 3 pm travelling at 60 miles per hour towards a "
       "city that is 150 miles away. At what time does it arrive? Answer: 5:30 pm.")
paraphrase = ("If a train departs at 3pm and moves at 60 mph toward a town 150 miles off, when will it get there?")
print("13-gram overlap with copied page:", len(ngrams(bench) & ngrams(web)),
      " with paraphrase:", len(ngrams(bench) & ngrams(paraphrase)))

# 16.7 temperature sampling over domains (or languages)
tokens = {"web_en": 10_000, "web_other": 2_000, "code": 800, "math": 100, "books": 300}   # billions
n = np.array(list(tokens.values()), dtype=float)
for T in (1, 2, 5):
    p = n ** (1 / T); p /= p.sum()
    print(f"T={T}: " + ", ".join(f"{k} {v:.1%}" for k, v in zip(tokens, p)))
budget = 15_000
p = n ** 0.5; p /= p.sum()
print("epochs per domain for a 15T-token run at T=2:", {k: round(float(budget * pi / ni), 1) for k, ni, pi in zip(tokens, n, p)})

# 16.1 storage arithmetic
print("15T tokens stored as token ids: %.0f TB as uint32, %.0f TB as uint16 (vocab < 65,536)" % (15e12 * 4 / 1e12, 15e12 * 2 / 1e12))
