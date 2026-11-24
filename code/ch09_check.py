import re, unicodedata, collections

def train_bpe(words, num_merges):
    vocab = collections.Counter(tuple(w) + ("</w>",) for w in words)
    merges = []
    for _ in range(num_merges):
        pairs = collections.Counter()
        for sym, f in vocab.items():
            for a, b in zip(sym, sym[1:]): pairs[(a, b)] += f
        if not pairs: break
        best = max(pairs, key=pairs.get); merges.append(best)
        new = collections.Counter()
        for sym, f in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new[tuple(out)] += f
        vocab = new
    return merges

def encode(word, merges):
    sym = list(word) + ["</w>"]
    for a, b in merges:
        i, out = 0, []
        while i < len(sym):
            if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(sym[i]); i += 1
        sym = out
    return sym

corpus = ("low lower lowest newer newest wider widest " * 5 + "order ordered ordering reorder reordered " * 8).split()
merges = train_bpe(corpus, 20)
print(merges[:6])
print(encode("reordering", merges), encode("lowered", merges))

def normalise_ar(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ً-ْـ]", "", s)          # harakat + tatweel
    s = re.sub(r"[آأإ]", "ا", s)     # alef variants -> bare alef
    s = s.replace("ى", "ي").replace("ة", "ه")  # alef maqsura -> ya, ta marbuta -> ha
    return s
print(normalise_ar("شَاوَرْمَا"), normalise_ar("أكلة"), normalise_ar("ﻻ"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.normalizer = normalizers.NFKC()
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["[UNK]", "[PAD]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
texts = ["chicken shawarma wrap with garlic sauce", "شاورما دجاج مع صوص الثوم", "spicy chicken burger", "burger lahm with cheese", "شاورما لحم"] * 200
tok.train_from_iterator(texts, trainer)
for t in ["chicken shawarma", "شاورما دجاج", "shawerma 3al fa7m"]:
    e = tok.encode(t); print(t, len(e.ids), e.tokens[:8])
print(len("شاورما دجاج".encode("utf-8")), len("chicken shawarma".encode("utf-8")))

# ---- second-edition worked example: the first BPE merges with their pair counts --------------
import collections as _c
words = ("low lower lowest newer newest wider widest " * 5 + "order ordered ordering reorder reordered " * 8).split()
vocab = _c.Counter(tuple(w) + ("</w>",) for w in words)
for step in range(4):
    pairs = _c.Counter()
    for sym, f in vocab.items():
        for a, b in zip(sym, sym[1:]): pairs[(a, b)] += f
    best, cnt = pairs.most_common(1)[0]
    runner = pairs.most_common(3)[1:]
    print(f"merge {step+1}: {best} count {cnt}; next best {runner}")
    new = _c.Counter()
    for sym, f in vocab.items():
        out, i = [], 0
        while i < len(sym):
            if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best: out.append(sym[i] + sym[i + 1]); i += 2
            else: out.append(sym[i]); i += 1
        new[tuple(out)] += f
    vocab = new

# ---- review additions ---------------------------------------------------------------------
import math

# 9.2 attention vs total compute when 1,000 characters become 250 tokens (d = 4,096)
d = 4096
def fwd_flops(T):                    # per sequence, per layer: linear layers 24 d^2 per token, attention 4 T d per token
    return T * 24 * d * d + 4 * T * T * d
print("attention-only ratio 1000 vs 250:", (1000 / 250) ** 2, " total ratio:", round(fwd_flops(1000) / fwd_flops(250), 2),
      " attention share at T=1000:", round(4 * 1000 * 1000 * d / fwd_flops(1000), 3))

# 9.4 Unigram: Viterbi segmentation with a toy piece vocabulary
probs = {"re": 0.05, "order": 0.04, "ing": 0.05, "reorder": 0.01, "or": 0.02, "der": 0.02,
         "in": 0.03, "g": 0.01, "r": 0.01, "e": 0.02, "o": 0.01, "d": 0.01, "i": 0.01, "n": 0.01}
def viterbi(word):
    best = [(0.0, None)] + [(-math.inf, None)] * len(word)
    for end in range(1, len(word) + 1):
        for start in range(end):
            piece = word[start:end]
            if piece in probs and best[start][0] > -math.inf:
                sc = best[start][0] + math.log(probs[piece])
                if sc > best[end][0]: best[end] = (sc, start)
    pieces, end = [], len(word)
    while end > 0:
        start = best[end][1]; pieces.append(word[start:end]); end = start
    return pieces[::-1], best[-1][0]
seg, lp = viterbi("reordering")
print("Viterbi:", seg, "prob", f"{math.exp(lp):.1e}")
for alt in (["re", "order", "ing"], ["re", "or", "der", "ing"]):
    print("  alternative", alt, f"{math.prod(probs[p] for p in alt):.1e}")

# 9.4 byte-level BPE: GPT-2's bytes-to-unicode map (space -> 'Ġ', newline -> 'Ċ')
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))
b2u = bytes_to_unicode()
print("byte 32 ->", b2u[32], hex(ord(b2u[32])), " byte 10 ->", b2u[10], " 256 distinct:", len(set(b2u.values())) == 256)

# production-style encoding: merge the lowest-rank pair first; same result as replaying the merge list
rank = {m: i for i, m in enumerate(merges)}
def encode_by_rank(word):
    sym = list(word) + ["</w>"]
    while True:
        pairs = [(rank.get((a, b), math.inf), i) for i, (a, b) in enumerate(zip(sym, sym[1:]))]
        r, i = min(pairs, default=(math.inf, -1))
        if r == math.inf: return sym
        a, b = sym[i], sym[i + 1]; out, j = [], 0
        while j < len(sym):
            if j < len(sym) - 1 and sym[j] == a and sym[j + 1] == b: out.append(a + b); j += 2
            else: out.append(sym[j]); j += 1
        sym = out
test_words = ["reordering", "lowered", "newest", "order", "widow", "lowness", "reorders"]
print("rank-based == replay for", len(test_words), "words:", all(encode_by_rank(w) == encode(w, merges) for w in test_words))

# 9.5 vocabulary parameters and output-layer compute
print("128,256 x 4,096 =", round(128256 * 4096 / 1e9, 3), "B params per matrix;",
      " output-layer FLOPs per token", round(2 * 128256 * 4096 / 1e9, 2), "GFLOP =",
      round(100 * 2 * 128256 * 4096 / (2 * 8e9), 1), "% of 2N for an 8B model")

# 9.7 cost with cached input and reasoning tokens (placeholder prices)
N = 3e6; p_in, p_cache, p_out = 0.50, 0.05, 1.50
naive = N * (400 * p_in + 60 * p_out) / 1e6
cached = N * (150 * p_in + 250 * p_cache + 60 * p_out) / 1e6
reasoning = N * (150 * p_in + 250 * p_cache + (60 + 500) * p_out) / 1e6
print(f"review summaries: naive ${naive:.0f}, with cached prefix ${cached:.1f}, batch 50% off ${cached / 2:.1f}, "
      f"with 500 reasoning tokens ${reasoning:.1f}")

# 9.8 tokenisation seams: encode prompt and completion separately vs jointly
seam_texts = ["The capital of France is Paris.", "We flew to Paris in May.", "Paris is large.", "The answer is 42.",
              "The capital of Italy is Rome.", "Rome and Paris are capitals."] * 300
st = Tokenizer(models.BPE()); st.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
st.train_from_iterator(seam_texts, trainers.BpeTrainer(vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
prompt, completion = "The capital of France is ", "Paris"
sep = st.encode(prompt).tokens + st.encode(completion).tokens
joint = st.encode(prompt + completion).tokens
print("separately:", sep[-3:], " jointly:", joint[-3:], " same ids:", sep == joint)

# 9.11 bits per byte from per-token loss
for name, loss, bpt in (("A", 2.0, 4.0), ("B", 1.8, 3.2)):
    print(f"model {name}: {loss} nats/token, {bpt} bytes/token -> perplexity {math.exp(loss):.2f}, bits/byte {loss / (bpt * math.log(2)):.3f}")
