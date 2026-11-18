# 09 · Text and tokenisation

## Why it matters

Every LLM bill, context-window limit and many strange model failures trace back to the tokenizer. A model never sees letters: it sees integer ids from a fixed vocabulary chosen before pre-training. That choice decides how many tokens your Arabic menu costs, why the model cannot count the r's in "strawberry", why prices like "12.50 AED" are split oddly, and why a fine-tuned model breaks if you change the chat template. In a region where users mix Arabic, English and Arabizi ("3al fa7m") in one query, tokenisation and normalisation are real product questions, not trivia.

## Key concepts

- **Token and vocabulary**: models read integer ids from a fixed vocabulary of subword pieces, not characters or words. [Hugging Face course: the Tokenizers library](https://huggingface.co/learn/llm-course/chapter6/1).
- **Byte-pair encoding (BPE)**: build a vocabulary by repeatedly merging the most frequent adjacent pair. [BPE for NMT](https://arxiv.org/abs/1508.07909).
- **WordPiece and Unigram**: alternatives that choose merges by likelihood, or prune a large vocabulary. [Subword Regularization (Unigram)](https://arxiv.org/abs/1804.10959).
- **SentencePiece and byte-level BPE**: tokenisers that work on raw text or raw bytes, so any language and any character can be encoded. [SentencePiece](https://arxiv.org/abs/1808.06226).
- **Normalisation**: Unicode normalisation, case and accent handling, and script-specific cleaning before tokenisation.
- **Fertility**: tokens per word; higher for languages under-represented in the tokenizer's training data, which raises cost and lowers quality.
- **Token counting and cost**: always count with the model's own tokenizer. [tiktoken](https://github.com/openai/tiktoken) for OpenAI models; Hugging Face tokenizers for open models.
- **Build one**: Karpathy's "Let's build the GPT Tokenizer" in [Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## 9.1 The pipeline

![Diagram](images/09-01.png)

*A tokenizer is a pipeline. Each stage is a choice baked into the model at pre-training time.*

- **Normalisation**: Unicode normalisation (NFC/NFKC), optional lower-casing and accent stripping. A cased model must not get lower-cased input.
- **Pre-tokenisation**: split into rough words on spaces and punctuation, or treat the raw bytes as the base alphabet (GPT-2 style byte-level, where a space becomes the marker `Ġ`).
- **Subword model**: split each piece into vocabulary units.
- **Post-processing**: add special tokens (`[CLS]` / `[SEP]`, BOS/EOS) and, for chat models, the role markers of the chat template.

Always load the tokenizer that matches the model checkpoint. Mixing them gives no error, just garbage.

## 9.2 Why subwords

Word-level vocabularies explode in size and still meet unknown words (new dishes, brand names, typos). Character-level sequences are very long, which is costly with attention's quadratic cost. Subwords sit between: frequent words stay whole, rare words split into reusable pieces, and nothing is ever out of vocabulary if the base alphabet covers all bytes.

## 9.3 BPE from scratch

Byte-pair encoding (Sennrich et al., 2016) starts from characters and repeatedly merges the most frequent adjacent pair. The list of merges, in order, *is* the tokenizer.

```python
import collections

def train_bpe(words, num_merges):
    vocab = collections.Counter(tuple(w) + ("</w>",) for w in words)
    merges = []
    for _ in range(num_merges):
        pairs = collections.Counter()
        for sym, f in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        new = collections.Counter()
        for sym, f in vocab.items():       # apply the merge everywhere
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new[tuple(out)] += f
        vocab = new
    return merges

def encode(word, merges):                  # replay merges in training order
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

corpus = ("low lower lowest newer newest wider widest " * 5
          + "order ordered ordering reorder reordered " * 8).split()
merges = train_bpe(corpus, 20)
# merges[:6] -> [('e','r'), ('d','er'), ('o','r'), ('or','der'), ('order','</w>'), ('order','e')]
# encode("reordering") -> ['re', 'orderin', 'g', '</w>']   (never seen, still encoded)
# encode("lowered")    -> ['low', 'er', 'e', 'd', '</w>']
```

This naive version is O(merges × corpus); real implementations keep pair counts updated incrementally and are written in Rust. In an interview, writing this cleanly in 15 minutes is a realistic ask.

## 9.4 The other algorithms

- **WordPiece** (BERT): like BPE, but picks the merge that most increases the likelihood of the training data, roughly `count(ab) / (count(a)·count(b))`, so it prefers pairs that are surprising together rather than just frequent. Continuation pieces carry a `##` prefix (`order ##ing`). Encoding is greedy longest-match.
- **Unigram LM** (Kudo, 2018): start from a big candidate vocabulary and prune pieces whose removal hurts the likelihood least. Encoding picks the most probable segmentation (Viterbi), and it can sample alternative segmentations during training (subword regularisation) for robustness to typos.
- **SentencePiece**: a library, not an algorithm. It runs BPE or Unigram on raw text with spaces kept as a symbol (`▁`), so no language-specific pre-tokeniser is needed and decoding is lossless. Used by T5, Llama 1–2 and many multilingual models.
- **Byte-level BPE** (GPT-2 onwards, tiktoken): the base alphabet is the 256 bytes, so any string, emoji or script is encodable without an unknown token.

Training your own with Hugging Face `tokenizers` takes a few lines:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.normalizer = normalizers.NFKC()
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["[UNK]", "[PAD]"],

initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(menu_texts, trainer)

tok.encode("chicken shawarma").tokens   # ['chicken', 'Ġshawarma']   2 tokens
tok.encode("shawerma 3al fa7m")          # 13 tokens: the misspelt Arabizi breaks into pieces
```

The toy vocabulary was trained on a few repeated English and Arabic menu lines, so both seen phrases became 2 tokens, while unseen Arabizi fell apart into 13. Byte-level tokens print as mojibake (`Ø´Ø§...`) for Arabic: that is the byte display, not corruption.

## 9.5 Vocabulary size trade-offs

| Larger vocabulary | Smaller vocabulary |
|---|---|
| Shorter sequences: cheaper attention, more text per context window | Longer sequences |
| Bigger embedding and output matrices (V × d each) | Fewer parameters spent on the vocabulary |
| Rare tokens get few training updates | Each token is well trained |
| Better coverage of many languages | Non-dominant languages fragment more |

Typical sizes: BERT about 30k, GPT-2 about 50k, recent multilingual LLMs 100k to 250k. Bigger vocabularies became popular partly to treat non-English languages more fairly.

## 9.6 Multilingual text, Arabic and code-switching

A tokenizer trained mostly on English splits other scripts into more pieces. The same meaning then costs more tokens, fills the context faster, and is often modelled worse. In UTF-8 an Arabic letter takes 2 bytes against 1 for a Latin letter ("دجاج شاورما" is 21 bytes vs 16 for "chicken shawarma"), which hurts byte-level tokenizers with little Arabic in training. Measure your own token-per-word ratio per language before quoting costs.

Arabic-specific normalisation that helps search and classical NLP (be careful with generative models, which should usually see text as written):

```python
import re, unicodedata

def normalise_ar(s):
    s = unicodedata.normalize("NFKC", s)                 # also splits ligatures: "ال" >- "ﻻ"
    s = re.sub(r"[ً-ْـ]", "", s)         # diacritics (harakat) and tatweel
    s = re.sub(r"[آأإ]", "ا", s)    # alef variants -> bare alef
    s = s.replace("ي" ,"ى").replace("ه" ,"ة")  # alef maqsura, ta marbuta
    return s

normalise_ar("َشاَوْرَما")   # -> "شاورما"
normalise_ar("أكلة")     # -> "اكله"
```

Code-switching and Arabizi (Arabic in Latin letters, with digits for sounds: 3 = = 2 ,ح = 7 ,ع ء) are common in Gulf search queries and reviews. Options, in order of cost: character n-gram features (fastText, Chapter 8) that tolerate spelling variants; a transliteration step for the most frequent terms; a multilingual embedding model evaluated on your own query logs; or fine-tuning on labelled mixed-language data. Always split your evaluation set by language and script so a good English score cannot hide a poor Arabic one.

## 9.7 Tokens and money

API prices are per token, usually different for input and output, so cost is arithmetic you should do on a whiteboard:

```text
cost = N_requests × (tokens_in × price_in + tokens_out × price_out) / 1e6

Example (prices are placeholders, check the provider's page):
2,000,000 merchant descriptions × (400 in, 20 out)
price_in = $0.50 / 1M, price_out = $1.50 / 1M
= 2e6 × (400×0.50 + 20×1.50) / 1e6 = 2 × (200 + 30) = $460
Batch API at 50% off -> $230; cache the shared system prompt -> lower still.
```

Levers, roughly in order of impact: a smaller model for the easy cases (cascade, Chapter 14), shorter prompts (move fixed instructions into a cached prefix), structured short outputs instead of prose, batch APIs for anything not latency-sensitive, and fewer calls (dedupe identical inputs). Combined, these levers often cut an offline LLM job's cost by 5–10×. If you have done this, have the before and after numbers ready and know which lever gave how much.

Count tokens with the model's own tokenizer (tiktoken for OpenAI models, the provider's token-count endpoint, or the Hugging Face tokenizer for open models). Character-based rules of thumb (about 4 English characters per token) do not hold for Arabic or code.

## 9.8 Tokenisation quirks behind model failures

- **Counting and spelling**: "strawberry" may be 2–3 tokens, so the model never sees individual letters.
- **Numbers**: digits group inconsistently ("1234" vs "12" + "34"), which hurts arithmetic; some newer tokenizers split numbers into single digits for this reason.
- **Leading spaces**: "Dubai" and " Dubai" are different tokens; prompt templates that add or drop a space change behaviour.
- **Trailing whitespace** at the end of a prompt can push the model into an unusual token and degrade output.
- **Glitch tokens**: tokens that were in the vocabulary but almost never in training data give erratic behaviour.
- **Chat templates**: fine-tuning with one template and serving with another is a silent quality killer. Use `tokenizer.apply_chat_template`.

## Pitfalls

- Using a different tokenizer from the model's (for example a cased model with uncased input).
- Truncating at a character limit instead of a token limit, or truncating away the question at the end of a long RAG prompt.
- Estimating cost from English-only samples when traffic is half Arabic.
- Normalising away diacritics or case for a generation model that needs them.
- Adding new special tokens without resizing the embedding matrix (`model.resize_token_embeddings`).

## Interview questions

**Mid level**

1. Why do modern models use subword tokenisation?
2. Walk through how BPE is trained and how it encodes a new word.
3. What is the difference between BPE and WordPiece? What does SentencePiece add?
4. Why can LLMs struggle to count letters in a word?

**Senior level**

5. Your Arabic queries cost far more per request than English ones. Why, and what can you do?
6. You need to classify search queries that mix Arabic, English and Arabizi. Design the text-processing part.
7. Estimate the monthly cost of summarising every restaurant review with an LLM, and say how you would cut it in half.
8. What are the trade-offs of a larger vocabulary?

Short answers: (5) the tokenizer was trained on mostly English text, so Arabic splits into more tokens; measure it, prefer models with better multilingual tokenizers, shorten prompts, cache shared prefixes, consider a smaller fine-tuned model. (6) Unicode normalisation; script detection; light Arabic normalisation for retrieval features; character n-gram or multilingual embedding model; a transliteration dictionary for the top Arabizi terms from logs; evaluation split by script. (7) Reviews per month × average tokens in and out × prices; halve it with a batch API, a smaller model for short reviews, summarising only reviews that are shown, and caching instructions.

## Practice

- Write `train_bpe` and `encode` from a blank file. Then add a decoder that turns tokens back into the string.
- Take 200 real menu items or queries in Arabic and English and count tokens with two different model tokenizers. Report tokens per word by language.
- Train a 5k-vocab tokenizer on a public multilingual corpus with `tokenizers`, and compare fertility (tokens per word) with a pre-trained one.
- Redo the cost arithmetic for a project of your own (or the review-summary example above) with current prices.

## Read next

- Sennrich, Haddow and Birch, "Neural Machine Translation of Rare Words with Subword Units" (2016).
- Kudo, "Subword Regularization" (2018), and Kudo and Richardson, "SentencePiece" (2018).
- Andrej Karpathy, "Let's build the GPT Tokenizer" (video, 2024).
- Hugging Face course, the Tokenizers chapter.
- Rust et al., "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models" (2021).
- Links to the papers, course and tools are in the Key concepts section at the top of this chapter.

## Done when

You can implement BPE training and encoding from scratch. You can explain BPE, WordPiece, Unigram and byte-level in one sentence each.

You can do LLM cost arithmetic out loud and name five ways to cut it. You can explain why Arabic and code-switched text need care, and what you would do about it.

You can list three model failures caused by tokenisation.

---

[← 08 · Embeddings and sequence models](08-embeddings-and-sequence-models.md) · [Contents](../README.md#contents) · [10 · The transformer in depth →](10-the-transformer-in-depth.md)
