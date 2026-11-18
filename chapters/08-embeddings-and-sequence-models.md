# 8 · Embeddings and sequence models

## Why it matters

Transformers did not appear from nowhere. Word embeddings taught the field that meaning can live in dense vectors; RNNs and LSTMs taught it how to read sequences; and attention was invented to fix a weakness of RNN encoder-decoders. Interviewers still ask about these because they reveal whether you understand *why* the transformer looks the way it does. In practice you will also meet them: fastText and word2vec-style item embeddings ("item2vec") are still common in recommendation, including the order-history models in Chapter 25, and small GRUs still run on-device and in streaming systems.

Code below was run with PyTorch on CPU; quoted outputs are from that run.

## Key concepts

- **Distributional hypothesis**: words that appear in similar contexts have similar meanings; every embedding method in this chapter builds on it.
- **word2vec (skip-gram, CBOW, negative sampling)**: learn word vectors by predicting context words. [word2vec paper](https://arxiv.org/abs/1310.4546), [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/).
- **GloVe**: word vectors fitted to global co-occurrence counts. [GloVe project page](https://nlp.stanford.edu/projects/glove/).
- **fastText**: word vectors built from character n-grams, so unseen and misspelt words still get a vector. [fastText paper](https://arxiv.org/abs/1607.04606).
- **RNN, LSTM, GRU**: networks that read a sequence step by step with a hidden state; gates in LSTM and GRU let information survive long sequences. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), [GRU paper](https://arxiv.org/abs/1406.1078).
- **Seq2seq**: an encoder reads the input and a decoder generates the output. [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215).
- **Attention (Bahdanau)**: each decoder step looks at all encoder states, weighted by relevance; the bridge to the transformer. [Bahdanau et al.](https://arxiv.org/abs/1409.0473).
- **Item2vec**: word2vec applied to sessions or baskets to embed products and restaurants.

## 8.1 From one-hot to dense vectors

A one-hot vector for a 50k vocabulary is 50k long, sparse, and says every pair of words is equally different. A learned embedding maps each id to a dense vector of, say, 300 dimensions where similar words end up close. `nn.Embedding(V, d)` is just a lookup table of shape `(V, d)`, mathematically the same as a one-hot vector times a weight matrix, but without building the one-hot.

The idea that makes them learnable is the **distributional hypothesis**: words that appear in similar contexts have similar meanings. Every method in this section is a different way to turn co-occurrence into vectors.

## 8.2 word2vec

Two variants (Mikolov et al., 2013):

- **Skip-gram**: from a centre word, predict each word in a window around it. Better for rare words.
- **CBOW**: from the average of the context words, predict the centre word. Faster.

The full softmax over the vocabulary is expensive, so the practical version uses **negative sampling** (SGNS): for each real (centre, context) pair, push their dot product up; for k random "noise" words, push it down. Noise words are drawn from unigram frequency raised to the 0.75 power, which samples rare words a bit more often than their raw frequency. Frequent words such as "the" are also subsampled.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SGNS(nn.Module):
    def __init__(self, V, d=32):
        super().__init__()
        self.inp = nn.Embedding(V, d)    # centre-word vectors (the ones you keep)
        self.out = nn.Embedding(V, d)    # context-word vectors
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d); nn.init.zeros_(self.out.weight)

    def forward(self, c, o, k=5):        # c, o: (B,) ids of centre and true context
        neg = torch.multinomial(noise, c.size(0) * k, replacement=True).view(-1, k)
        vc = self.inp(c)                                            # (B, d)
        pos = F.logsigmoid((vc * self.out(o)).sum(-1))              # (B,)
        negs = F.logsigmoid(-(self.out(neg) @ vc.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + negs).mean()
# noise = counts ** 0.75, normalised. Trained 3 epochs on a toy corpus of
# "i ordered <food> and <food> tonight" / "we flew to <city> from <city>":
# nearest("sushi") -> ['salad', 'burger', 'shawarma', 'pizza']
# nearest("dubai") -> ['cairo', 'amman', 'kuwait', 'doha']
```

The famous analogy result (king − man + woman ≈ queen) is real but cherry-picked; do not oversell it in interviews. What matters more: SGNS was later shown to be implicitly factorising a shifted PMI (pointwise mutual information) matrix (Levy and Goldberg, 2014), which links it to the count-based methods below.

**Item2vec.** Replace sentences with sessions or baskets and words with items: you get item embeddings for "customers who ordered X also ordered Y" and as features for a ranker. This is a cheap, strong baseline for food and grocery catalogues.

## 8.3 GloVe and fastText

- **GloVe** (Pennington et al., 2014) builds the global co-occurrence matrix X once, then fits vectors so that `w_i · w_j + b_i + b_j ≈ log X_ij`, with a weighting that caps very frequent pairs. Count-based and global, where word2vec is predictive and local; results are similar in practice.
- **fastText** (Bojanowski et al., 2017) represents each word as the sum of its character n-gram vectors (3 to 6 characters, plus the whole word). It can build a vector for a word it never saw, and it handles typos and rich morphology well. That matters for Arabic and for transliterated, code-switched Arabizi ("shawarma", "shawerma", "shwarma"), and for user-typed menu items. The fastText library also gives a very fast linear text classifier, still a good first baseline.

All three give **static** embeddings: one vector per word, whatever the context. "Bank" has the same vector in "river bank" and "bank account". Contextual embeddings from ELMo, then BERT (Chapter 11) fixed that.

## 8.4 Recurrent networks

An RNN reads a sequence one step at a time and keeps a hidden state: `h_t = tanh(W_x` `x_t + W_h h_{t-1} + b)`. The same weights are used at every step, so it handles any length.

The problem is training. Backpropagation through time multiplies by `W_h` (and the tanh derivative) once per step, so gradients over long distances shrink towards zero (**vanishing**) or blow up (**exploding**). Exploding is fixed with gradient clipping; vanishing needs a different cell.

**LSTM** (Hochreiter and Schmidhuber, 1997) adds a cell state `c_t` that is updated additively, controlled by gates:

- forget gate `f_t = σ(W_f [h_{t-1}, x_t])`: how much of the old cell to keep;
- input gate `i_t` and candidate `g_t = tanh(…)`: what to write;
- `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`;
- output gate `o_t`, and `h_t = o_t ⊙ tanh(c_t)`.

Because `c_t` is a sum rather than a repeated matrix product, gradients can flow across many steps when the forget gate stays near 1 (the same idea as residual connections in Chapter 6). **GRU** (Cho et al., 2014) merges the gates into two (update and reset) and drops the separate cell: fewer parameters, usually similar accuracy.

```python
class LSTMClassifier(nn.Module):
    def __init__(self, V, d=32, h=64, n_cls=2):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * h, n_cls)

    def forward(self, x, lengths):       # x: (B, T) padded ids, lengths: (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (2 directions, B, h)
        return self.head(torch.cat([h_n[0], h_n[1]], dim=-1))
```

Packing stops the LSTM from reading padding. On a toy task where the label is "does token 5 appear before token 7", this model reached **0.918** test accuracy after 8 epochs, while a mean-of-embeddings model (`nn.EmbeddingBag`) stayed at **0.49**, chance level. Order-blind models cannot learn order: the point of sequence models in one experiment.

## 8.5 Seq2seq and attention

The encoder-decoder (Sutskever et al., 2014) reads the input with one RNN, squeezes it into the final hidden state, and a second RNN generates the output from that single vector. It worked for translation, but quality fell on long sentences: one fixed-size vector is a bottleneck.

**Bahdanau attention** (2014) removes the bottleneck. At each output step the decoder state is used as a query, scored against every encoder state, softmaxed into weights, and the weighted sum (the context vector) is fed into the decoder.

![Diagram](images/08-01.png)

*Bahdanau attention. The decoder state is the query; the weights α decide how much of each encoder state goes into the context vector.*

```python
class AdditiveAttention(nn.Module):
    def __init__(self, d_dec, d_enc, d_att=64):
        super().__init__()
        self.W_q = nn.Linear(d_dec, d_att, bias=False)
        self.W_k = nn.Linear(d_enc, d_att, bias=False)
        self.v = nn.Linear(d_att, 1, bias=False)

    def forward(self, s, H, mask):   # s: (B, d_dec), H: (B, T, d_enc), mask: (B, T) True = real
        scores = self.v(torch.tanh(self.W_q(s).unsqueeze(1) + self.W_k(H))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = scores.softmax(-1)
        return (alpha.unsqueeze(-1) * H).sum(1), alpha    # context (B, d_enc), weights (B, T)
# masked positions get weight 0.0; each row of alpha sums to 1.0
```

**Luong attention** (2015) replaced the small network with a dot product `s · h_i` (or `sᵀ W` `h_i`): cheaper and just as good. Scaled dot-product attention in the transformer is that idea plus a `1/√d` scale, many heads, and, crucially, self-attention: every token attends to every other token in the same sequence. Once attention could do the job of recurrence, Vaswani et al. (2017) dropped recurrence entirely, which made training parallel across positions. Chapter 10 picks up from here.

**Teacher forcing.** During training, the decoder is fed the true previous token rather than its own prediction. Fast and stable, but at inference the model sees its own mistakes, a mismatch called exposure bias. Decoding strategies (Chapter 14) and sequence-level training are the usual answers.

## 8.6 Where these models still earn their place

- **Static embeddings** as cheap features, for very low-latency lookups, and for search query expansion.
- **Item2vec / session embeddings** for recommendations and basket completion.
- **GRU/LSTM** for streaming signals (sensor, rider GPS traces, event sequences) where you process one step at a time with constant memory, and on-device models.
- **fastText classifiers** as a 10-minute baseline before any transformer. If a fine-tuned transformer only beats fastText by one point, ask whether the extra cost is worth it.

## Pitfalls

- Comparing embeddings with Euclidean distance when they were trained with dot products; normalise and use cosine unless you know otherwise.
- Treating word2vec analogies as proof of quality. Evaluate on the downstream task.
- Feeding padded batches to an RNN without packing or masking, so the final hidden state is computed after reading padding.
- Using the last time step of a padded batch as "the sentence representation": for short sequences it is a padding step.
- Forgetting gradient clipping with RNNs.
- Assuming static embeddings handle polysemy or negation ("not good" vs "good").

## Interview questions

**Mid level**

1. Explain skip-gram with negative sampling. Why sample noise words from frequency to the 0.75?
2. What problem does fastText solve that word2vec does not?
3. Why do vanilla RNNs struggle with long sequences? How do LSTMs help?
4. What is the difference between an LSTM and a GRU?

**Senior level**

5. Explain the seq2seq bottleneck and how attention fixes it. How does Bahdanau attention relate to transformer attention?
6. You have 2 million order sessions and need item embeddings for a "you may also like" carousel tomorrow. What do you build first and how do you evaluate it?
7. What is exposure bias and how would you reduce it?
8. When would you still choose an RNN over a transformer in production?

Short answers: (1) It turns a V-way softmax into k+1 binary classifications; the 0.75 power flattens the distribution so rare words appear as negatives more often than their raw frequency would allow. (5) The encoder compresses everything into one vector; attention lets every decoder step look at all encoder states, weighted by relevance. The transformer keeps the query-key-value weighting, makes it multi-head and scaled dot-product, and applies it within the same sequence instead of recurrence. (6) Item2vec on sessions (items as words, sessions as sentences), with popular items subsampled; evaluate offline with next-item hit rate@k against a co-occurrence baseline, then an A/B test on carousel CTR and orders. (8) Streaming with constant memory per step, tiny on-device models, very long signals where quadratic attention is too costly (though state-space models now compete there).

## Practice

- Implement SGNS from the code above on your own order data or a public dataset (for example Instacart baskets). Inspect nearest neighbours for 20 items.
- Train fastText on a product-title classification task and compare with a TF-IDF + logistic regression baseline.
- Reproduce the order experiment: LSTM vs bag of embeddings vs a 1-layer transformer encoder. Plot accuracy against sequence length.
- Write additive attention from a blank file, with masking, and test that padded positions get zero weight.

## Read next

- Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (2013).
- Pennington, Socher and Manning, "GloVe: Global Vectors for Word Representation" (2014).
- Bojanowski et al., "Enriching Word Vectors with Subword Information" (2017).
- Christopher Olah, "Understanding LSTM Networks" (blog post, 2015).
- Bahdanau, Cho and Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014).
- Levy and Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (2014).
- Barkan and Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative Filtering" (2016).
- Links to the papers and explainers are in the Key concepts section at the top of this chapter.

## Done when

You can explain skip-gram, negative sampling and the 0.75 power without notes.

You can say how word2vec, GloVe and fastText differ, and when fastText wins.

You can write the LSTM gate equations and explain why they ease vanishing gradients.

You can implement additive attention with masking from scratch.

You can tell the story from seq2seq to the transformer in two minutes.

---

[← 07 · PyTorch for practitioners](07-pytorch-for-practitioners.md) · [Contents](../README.md#contents) · [09 · Text and tokenisation →](09-text-and-tokenisation.md)
