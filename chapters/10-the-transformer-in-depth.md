# 10 · The transformer in depth

Part III · Transformers and architectures. Assumes Chapter 5 (neural network basics) and Chapter 6 (PyTorch); Chapter 8 explains the attention mechanism that came before. Chapter 12 covers model families built on this architecture, Chapter 21 how they are served.

## Why it matters

Every LLM, embedding model and modern re-ranker you will use is a transformer. For an LLM-focused senior role, interviewers expect you to explain attention precisely, write it from scratch, and reason about its costs: why context length is expensive, what the KV cache is, why GQA and FlashAttention exist. These details also explain practical behaviour you meet daily: latency, memory limits, cost per token, and why long prompts degrade.

## Key concepts

- **The transformer**: an architecture built from attention and feed-forward blocks with residual connections, which replaced recurrence and made training parallel. [Attention Is All You Need](https://arxiv.org/abs/1706.03762); the best visual walkthrough is [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
- **Self-attention (queries, keys, values)**: each token builds its new representation as a weighted mix of all tokens, with weights from query–key similarity scaled by √d.
- **Multi-head attention**: several attention operations in parallel, each free to focus on a different kind of relationship.
- **Positional information (sinusoidal, learned, RoPE)**: attention has no notion of order, so position is injected. [RoPE paper](https://arxiv.org/abs/2104.09864).
- **Causal mask**: in decoders, each token may only attend to earlier tokens, which is what makes next-token training possible.
- **Feed-forward block, residual stream and layer norm (pre-norm)**: the per-token MLP, the additive path that carries information through the layers, and the normalisation that keeps training stable.
- **Efficient attention (FlashAttention)**: exact attention computed in tiles to save memory and time. [FlashAttention](https://arxiv.org/abs/2205.14135).
- **Build one**: [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (code with the paper) and Karpathy's "Let's build GPT" in [Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## 10.1 The big picture

A decoder-only language model (GPT, Llama, Claude-style models) turns a sequence of tokens into a probability distribution over the next token. It is built from one block repeated N times:

![Diagram](images/10-01.png)

*decoder-only transformer · pre-norm block, repeated*

1. **Embed** each token as a vector and add information about its position.
2. **Attention:** each position gathers information from earlier positions, weighted by relevance.
3. **Feed-forward:** each position is transformed independently by an MLP.
4. Both sub-layers use **residual connections** (add the input back) and **normalisation**.
5. After N blocks, a final linear layer maps each position's vector to scores over the vocabulary; softmax gives next-token probabilities.

Training is next-token prediction with cross-entropy over trillions of tokens; generation samples one token at a time and feeds it back in (Chapter 21).

## 10.2 Scaled dot-product attention

**Intuition.** Each position asks a question (its **query**), every position advertises what it contains (its **key**), and carries content to share (its **value**). A position's output is a weighted average of all values, with weights from how well its query matches each key. "it" in "the order arrived late because it was stuck" can attend strongly to "order".

**Maths.** With Q, K, V matrices (one row per position, created by learned linear projections of the input):

`Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ + M) V`

- Q Kᵀ gives every pair's similarity: a T × T matrix.
- **Dividing by √dₖ:** dot products of random dₖ-dimensional vectors have variance about dₖ; without scaling, large scores push softmax into saturation where gradients vanish.
- **M is the mask:** −∞ above the diagonal in a decoder (causal mask), so position t cannot see future positions; padding positions are masked too.
- Softmax turns each row into weights summing to 1; multiplying by V mixes the values.

```python
import numpy as np

def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)                      # (T, T) similarities
    if mask is not None:
        scores = np.where(mask, scores, -1e9)          # block future (or padded) positions
    w = np.exp(scores - scores.max(-1, keepdims=True)) # numerically stable softmax
    w /= w.sum(-1, keepdims=True)
    return w @ V, w

T, d = 4, 8
Q, K, V = (np.random.randn(T, d) for _ in range(3))
causal = np.tril(np.ones((T, T), dtype=bool))
out, weights = attention(Q, K, V, causal)              # row 1 has weights only on positions 0 and 1
```

**Self-attention vs cross-attention:** in self-attention Q, K and V come from the same sequence; in cross-attention (encoder-decoder models) queries come from the decoder and keys and values from the encoder output.

## 10.3 Multi-head attention

One attention pattern cannot capture everything (syntax, coreference, position, topic). Multi-head attention runs h attention operations in parallel on smaller projections (d_model / h dimensions each), concatenates the results, and mixes them with an output projection. Same cost as one big head; much more expressive.

The module below also supports **grouped-query attention** (fewer key/value heads than query heads, 10.9) and a **KV cache** for generation (10.8):

```python
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0):
        super().__init__()
        self.h, self.kv = n_heads, n_kv_heads or n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.k = nn.Linear(d_model, self.kv * self.dh, bias=False)
        self.v = nn.Linear(d_model, self.kv * self.dh, bias=False)
        self.o = nn.Linear(n_heads * self.dh, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)          # (B, h, T, dh)
        k = self.k(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        if kv_cache is not None:                                           # generation: reuse past keys/values
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        if self.kv != self.h:                                              # GQA: share each K/V head
            k = k.repeat_interleave(self.h // self.kv, dim=1)
            v = v.repeat_interleave(self.h // self.kv, dim=1)
        Tk = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        mask = torch.ones(T, Tk, dtype=torch.bool).tril(diagonal=Tk - T)   # causal, offset for the cache
        scores = scores.masked_fill(~mask, float("-inf"))
        att = F.dropout(F.softmax(scores, dim=-1), self.dropout, self.training)
        y = (att @ v).transpose(1, 2).reshape(B, T, self.h * self.dh)
        return self.o(y)
```

In practice you call `F.scaled_dot_product_attention`, which dispatches to fused kernels such as FlashAttention; writing it by hand is for understanding and interviews.

## 10.4 Positional encodings

Attention is permutation-invariant: without position information, "courier waited for customer" and "customer waited for courier" look the same. Options:

| Method | How | Notes |
|---|---|---|
| Sinusoidal (original transformer) | Add fixed sin/cos waves of different frequencies to embeddings | No parameters; extrapolates poorly in practice |
| Learned absolute (GPT-2, BERT) | Add a learned vector per position | Simple; cannot go beyond the trained length |
| RoPE (rotary, RoFormer; Llama and most current LLMs) | Rotate query and key vectors by an angle proportional to position | Attention scores depend on relative position; extended to longer contexts with scaling tricks (position interpolation, NTK-aware scaling, YaRN) |
| ALiBi | Add a penalty to attention scores proportional to distance | No embeddings; loss stays stable beyond the training length, though distant context is used weakly |

**RoPE in code,** and its key property: the dot product between a rotated query at position m and a rotated key at position n depends only on m − n.

```python
def rope(x, base=10000.0):
    """x: (B, heads, T, D). Rotate pairs of dimensions by position-dependent angles."""
    B, H, T, D = x.shape
    half = D // 2
    freqs = base ** (-torch.arange(half, dtype=torch.float32) / half)       # one frequency per pair
    angles = torch.arange(T, dtype=torch.float32)[:, None] * freqs[None]    # (T, half)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# Same q and k vectors placed at every position:
# score(q at 3, k at 1) == score(q at 5, k at 3)   -> both 1.4613 in a test run: only the distance matters
```

(This uses the "half split" pairing used by Llama-style code; the original paper pairs adjacent dimensions. Both are equivalent up to a permutation.)

## 10.5 Residual connections and normalisation

**Residuals** (x + sublayer(x)) give gradients a direct path through deep stacks and let each layer learn a small correction instead of a whole transformation. Without them, deep transformers do not train.

**Layer normalisation** normalises each position's vector to zero mean and unit variance across features (unlike batch norm, which normalises across the batch, a bad fit for variable-length sequences). Two placements:

- **Post-norm** (original transformer): Norm(x + sublayer(x)). Can reach good quality but needs careful learning-rate warm-up; unstable when deep.
- **Pre-norm** (GPT-2 onwards, nearly all modern LLMs): x + sublayer(Norm(x)). The residual path stays clean, training is far more stable, at some cost in final quality that other tricks recover.

**RMSNorm** (Llama and most current models) drops the mean-centring and bias, dividing only by the root mean square: cheaper, and works as well.

## 10.6 The feed-forward block and SwiGLU

Each block's MLP is applied to every position independently, usually expanding to about 4× the model width and back. It holds roughly two-thirds of a model's parameters and is thought to store much of its factual knowledge.

Modern LLMs use **gated** variants. SwiGLU multiplies a SiLU-activated projection by a second linear projection, element-wise, before projecting back (hidden size is reduced to about 8/3 × d to keep the parameter count equal):

```python
class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, d, bias=False)   # back to model width

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**Mixture of experts (MoE)** replaces the single FFN with many expert FFNs and a router that sends each token to a few of them, so total parameters grow while compute per token stays modest (Chapter 12).

## 10.7 Encoder, decoder and encoder-decoder

| Architecture | Attention | Training objective | Examples | Best for |
|---|---|---|---|---|
| Encoder-only | Bidirectional (sees left and right) | Masked language modelling | BERT, RoBERTa, modern embedding models | Classification, extraction, embeddings, re-rankers |
| Decoder-only | Causal (sees left only) | Next-token prediction | GPT family, Llama, Claude-style LLMs | Generation, chat, general-purpose LLMs |
| Encoder-decoder | Encoder bidirectional, decoder causal with cross-attention | Span corruption, seq2seq | T5, BART, many translation models | Translation, summarisation, structured transformation |

Decoder-only models won for general LLMs because one simple objective scales smoothly with data and compute, and every task can be phrased as text continuation. Encoders remain the efficient choice for classification and embeddings at high volume (Chapter 41).

## 10.8 Complexity and the KV cache

**Cost per layer** for a sequence of T tokens with width d: attention is O(T²·d) (every pair of positions) and the feed-forward block O(T·d²). For short sequences the FFN dominates; for long ones attention does, and attention memory grows with T² unless computed in tiles (FlashAttention).

**Generation and the KV cache.** A model generates one token at a time. Without caching, each new token would recompute keys and values for the whole prefix. The KV cache stores every layer's past keys and values, so each step only computes the new token's query, key and value and attends over the cache. In the mini-GPT below, logits from a cached incremental step match a full forward pass exactly.

The cache's memory is often the bottleneck of LLM serving:

`KV bytes = 2 (K and V) × layers × kv_heads × head_dim × sequence length ×` `batch × bytes per value`

```python
def kv_cache_gb(layers, kv_heads, head_dim, seq_len, batch=1, bytes_per_value=2):
    return 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_value / 1e9

kv_cache_gb(32, 32, 128, 8192)   # ~4.3 GB per sequence with full multi-head attention (fp16)
kv_cache_gb(32, 8, 128, 8192)    # ~1.1 GB with 8 KV heads (GQA): 4x more sequences per GPU
```

That is why serving engines manage the cache in pages (vLLM's PagedAttention), quantise it, and why architectures cut KV heads (Chapter 21).

## 10.9 Attention variants

| Variant | Idea | Why |
|---|---|---|
| Multi-query attention (MQA) | All query heads share one K/V head | Smallest KV cache, fastest decoding; some quality loss |
| Grouped-query attention (GQA) | Query heads share K/V heads in groups (for example 32 query, 8 KV) | Most of MQA's savings with near-full quality; standard in current open LLMs |
| FlashAttention | Exact attention computed in tiles in fast on-chip memory, never materialising the T×T matrix | Much faster and memory-linear in T; same result |
| Sliding-window attention | Each token attends only to the last w tokens | Linear cost; information still flows across layers; often mixed with some global layers |
| Sparse and linear attention, state-space models | Restrict or approximate attention, or replace it with recurrent state (Mamba-style) | Very long sequences; hybrids are an active research area |
| Multi-head latent attention (MLA) | Compress keys and values into a small latent vector cached per token (DeepSeek) | Very small KV cache |

## 10.10 A mini-GPT from scratch

Putting it together: a pre-norm decoder with learned positions, weight tying between the embedding and output layers, and cached generation with top-k sampling. Trained for 300 steps on a tiny repeated sentence at character level, it reaches a training loss below 0.1, enough to see the machinery work end to end.

```python
class Block(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)      # pre-norm residual
        return x + self.ff(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, n_layers=4, max_len=128):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight             # weight tying

    def forward(self, idx, caches=None, start=0):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(start, start + T))
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln(x))                   # logits (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        caches = [{} for _ in self.blocks]
        logits, out = self(idx, caches), idx            # prefill: process the prompt once
        for _ in range(n_new):                          # decode: one token per step
            last = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(last, top_k)
                last[last < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(last, -1), 1)
            out = torch.cat([out, nxt], 1)
            logits = self(nxt, caches, start=out.shape[1] - 1)
        return out

# Training step (xb, yb are input and next-token targets, shifted by one):
# loss = F.cross_entropy(model(xb).reshape(-1, vocab), yb.reshape(-1))
```

The two phases in `generate`, **prefill** (process the whole prompt in parallel) and **decode** (one token at a time, memory-bound on reading weights and the cache), are the two phases you optimise when serving (Chapter 21).

## Pitfalls

- **Forgetting the √dₖ scaling** or the causal mask when implementing attention.
- **Masking with 0 instead of −∞** before softmax (masked positions still get weight).
- **Mixing up which dimension softmax runs over** (over keys, the last dimension).
- **Assuming a model handles any context length** because RoPE is relative: quality degrades beyond the trained length without scaling and fine-tuning.
- **Ignoring KV-cache memory** when estimating serving capacity.
- **Thinking attention weights are explanations:** they show where information flowed in one layer, not why the model decided.
- **Treating encoder and decoder models as interchangeable** for embeddings or classification.

## Interview questions

Questions 1–5 basic, 6–11 intermediate, 12–15 senior.

**1. Explain self-attention.** Each position computes a query, key and value by linear projections; attention weights are the softmax of scaled query-key dot products; each output is the weighted sum of values. It lets every token gather information from relevant tokens anywhere in the (allowed) context.

**2. Why divide by √dₖ?** Dot products of dₖ-dimensional vectors have variance growing with dₖ; large scores saturate softmax and shrink gradients. Scaling keeps them in a trainable range.

**3. Why multiple heads?** Each head can attend to different relations (syntax, coreference, position) in its own subspace; concatenating them is more expressive than one head of the same total size, at the same cost.

**4. What is the causal mask?** A mask setting scores for future positions to −∞, so each token only attends to itself and earlier tokens; required for next-token training and generation.

**5. BERT vs GPT?** BERT: encoder, bidirectional attention, masked-token objective; best for understanding tasks and embeddings. GPT: decoder, causal attention, next-token objective; best for generation.

**6. Why do transformers need positional encodings, and how does RoPE work?** Attention ignores order. RoPE rotates queries and keys by position-dependent angles so their dot product depends on relative distance, which works naturally with caching; running beyond the trained length still needs scaling tricks (position interpolation, NTK, YaRN) and some long-context fine-tuning.

**7. Pre-norm vs post-norm?** Pre-norm normalises inside the residual branch, keeping the residual path an identity: much more stable training for deep models; post-norm needs careful warm-up.

**8. What is the KV cache and how big is it?** Stored keys and values of past tokens per layer, so decoding computes only the new token. Size: 2 × layers × KV heads × head dim × tokens × bytes. For a 32-layer model with 32 KV heads of 128 at 8k tokens in fp16, about 4.3 GB per sequence.

**9. What do MQA and GQA change?** They share key/value heads across query heads, shrinking the KV cache (and memory bandwidth per decoding step) by the sharing factor, allowing larger batches and longer contexts; GQA keeps quality close to full multi-head attention.

**10. What does FlashAttention do?** Computes exact attention in tiles that fit in on-chip memory, with an online softmax, never materialising the T×T matrix; reduces memory traffic, so it is faster and memory-linear in T.

**11. What is the computational complexity of a transformer layer?** Attention O(T²d), FFN O(Td²); attention dominates for long sequences, the FFN for short ones.

**12. Why did decoder-only models win for general-purpose LLMs?** One self-supervised objective on any text, smooth scaling with data and compute, every task expressible as continuation, simple efficient inference with KV caching, and instruction tuning on top.

**13. How do you extend a model's context length?** RoPE scaling (position interpolation, NTK-aware, YaRN) plus fine-tuning on long sequences; efficient attention kernels; GQA and cache quantisation for memory; then evaluate on realistic long-context tasks, not only needle tests.

**14. Why is decoding memory-bound while prefill is compute-bound?** Prefill processes many tokens per weight load (large matrix multiplications); decode processes one token per sequence per step, so each step reads all weights and the cache for little compute. Batching more sequences raises arithmetic intensity, which is why continuous batching matters.

**15. Walk me through what happens when an LLM generates one token.** The new token is embedded; in each layer, it is normalised, projected to Q, K, V (RoPE on Q, K), K and V appended to the cache, attention computed over the cache, added to the residual, then the FFN; after the last layer, normalisation and the output projection give logits; sampling (temperature, top-p) picks the token.

## Practice

1. **Attention from scratch** in NumPy with a causal mask, then in PyTorch; compare with `F.scaled_dot_product_attention` (`is_causal=True`).
2. **Multi-head and GQA:** implement the module above and verify output shapes and that GQA with kv = h equals standard attention.
3. **RoPE:** implement it and verify the relative-position property numerically.
4. **Mini-GPT:** train it on a small text corpus (Tiny Shakespeare) at character level; plot training and validation loss; generate samples with different temperatures.
5. **KV cache:** confirm cached and uncached generation give identical logits; time both for 200 tokens.
6. **Karpathy's "Let's build GPT"** video and nanoGPT repository: follow along, then rebuild without looking.
7. **Memory maths:** compute KV-cache and weight memory for three open models from their config files.

## Read next

- Vaswani et al., *Attention Is All You Need* (2017).
- Jay Alammar, *The Illustrated Transformer* (blog).
- Andrej Karpathy, *Neural Networks: Zero to Hero*, especially "Let's build GPT: from scratch, in code, spelled out", and the nanoGPT repository.
- *Dive into Deep Learning*: the chapters on attention mechanisms and transformers.
- *Build a Large Language Model (From Scratch)* (Raschka): chapters on attention and implementing a GPT model.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021); Press et al., *ALiBi* (2021).
- Ainslie et al., *GQA* (2023); Shazeer, *Fast Transformer Decoding: One Write-Head Is All* *You Need* (MQA, 2019).
- Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023).
- Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020).

**Links:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). Other titles above can be found by searching their title.

## Done when

You can write scaled dot-product attention with a causal mask from memory. You can implement multi-head attention with GQA and a KV cache. You can explain and implement RoPE and compare it with other positional encodings. You can explain pre-norm, RMSNorm, residuals and SwiGLU. You can compare encoder, decoder and encoder-decoder models and choose one. You can compute KV-cache memory and explain prefill vs decode. You can explain MQA, GQA, FlashAttention and sliding-window attention.

You have trained a mini-GPT yourself.

---

[← 09 · Text and tokenisation](09-text-and-tokenisation.md) · [Contents](../README.md#contents) · [11 · Model families and scaling →](11-model-families-and-scaling.md)
