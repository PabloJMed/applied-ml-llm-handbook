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
