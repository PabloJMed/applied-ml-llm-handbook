# 7 · Optimisation and training dynamics

Part II · Deep learning and training dynamics. Assumes Chapter 5 (backpropagation, initialisation, normalisation) and Chapter 6 (PyTorch training loops). Chapter 17 applies these ideas at the scale of a pretraining run; Chapter 13 uses them to explain why scaling laws can be fitted at all.

## Why it matters

A research scientist spends more time looking at loss curves than at anything else. Whether a run diverges at step 40,000, whether a new architecture "doesn't work" or was simply given the wrong learning rate, whether a 2× larger batch buys anything, whether fp8 is safe: these are optimisation questions, and they decide what is learned from every GPU-hour. Interviews for LLM roles probe them directly: "write AdamW", "why warm-up?", "what is µP for?", "the loss spiked, what do you do?". This chapter gives you the mechanics and the reasoning behind them, with numbers you can reproduce on a laptop.

Numbers in this chapter come from `code/ch07_check.py` (CPU, about two seconds).

## Key concepts

- **AdamW**: Adam with decoupled weight decay; the default optimiser for transformers. [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101).
- **Learning-rate schedules**: warm-up, cosine decay, and warmup-stable-decay (WSD), which keeps a constant rate and decays only at the end. [MiniCPM (WSD)](https://arxiv.org/abs/2404.06395).
- **Gradient noise scale and critical batch size**: the batch size beyond which larger batches stop saving steps. [McCandlish et al.](https://arxiv.org/abs/1812.06162).
- **Second-order-flavoured optimisers**: Adafactor, Lion, Shampoo, SOAP and Muon. [Adafactor](https://arxiv.org/abs/1804.04235), [Lion](https://arxiv.org/abs/2302.06675), [SOAP](https://arxiv.org/abs/2409.11321), [Muon is scalable (Moonlight)](https://arxiv.org/abs/2502.16982).
- **µP (maximal update parametrisation)**: scale initialisation and per-layer learning rates with width so the best hyperparameters transfer from small to large models. [Tensor Programs V](https://arxiv.org/abs/2203.03466).
- **Numerical formats**: fp32, bf16, fp16, fp8 (e4m3, e5m2) and microscaling (MX) formats. [FP8 formats for deep learning](https://arxiv.org/abs/2209.05433).
- **Stability tools**: gradient clipping, z-loss, QK-norm, logit soft-capping. [PaLM (z-loss)](https://arxiv.org/abs/2204.02311), [Scaling ViT to 22B (QK-norm)](https://arxiv.org/abs/2302.05442).
- **Edge of stability**: full-batch training drives the loss sharpness up to about 2/η and holds it there. [Cohen et al.](https://arxiv.org/abs/2103.00065).

## 7.1 From SGD to AdamW

SGD moves every parameter by the learning rate times its gradient. That is fragile for transformers: gradients for embeddings, attention projections and layer norms differ by orders of magnitude, so one learning rate is too large for some and too small for others.

**Adam** keeps two running averages per parameter: the mean of the gradient (momentum, β₁ ≈ 0.9) and the mean of its square (β₂ ≈ 0.95 for LLMs, 0.999 elsewhere). The update divides one by the square root of the other, so each coordinate moves by roughly the learning rate in units of its own typical gradient size. Bias correction undoes the zero initialisation of both averages in the first steps.

**AdamW** applies weight decay directly to the weights instead of adding it to the gradient. With Adam, an L2 term added to the gradient is divided by √v like everything else, so heavily-updated weights are barely regularised; decoupling fixes that.

```python
def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    p = p * (1 - lr * wd)                      # decoupled weight decay
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return p - lr * m_hat / (v_hat.sqrt() + eps), m, v
# matches torch.optim.AdamW to 1e-7 over three steps
# first step, gradients 1e-4 and 1e2: both coordinates move by exactly 0.001
```

The last line is the key property and the key risk. Adam's first step moves every coordinate by the learning rate, whatever its gradient. That is why Adam needs **warm-up**: at step one the second-moment estimate is built from a single gradient, so the normalised update is large and noisy in every direction at once.

Practical settings for LLM pretraining: β₁ = 0.9, β₂ = 0.95 (a shorter memory for v reacts faster to gradient spikes), ε = 1e-8 (some teams use smaller ε for very large models, because ε starts to dominate when gradients become tiny), weight decay 0.1 on matrices but not on norms, biases or often embeddings.

Memory: Adam stores two fp32 states per parameter, 8 bytes, on top of weights and gradients. That is the 16 bytes per parameter of mixed-precision training (Chapter 6) and the reason optimiser-state sharding (ZeRO, Chapter 17) exists.

## 7.2 Learning-rate schedules

Three phases appear in nearly every LLM run:

1. **Warm-up**: increase the rate linearly from near zero over the first few hundred to few thousand steps. It gives Adam's second moment time to settle and avoids a large, random first move.
2. **Stable or decaying phase**: most of the training.
3. **Final decay**: bring the rate down. The loss drops sharply when the rate falls, because the optimiser stops bouncing around the bottom of the valley.

**Cosine decay** (to about 10% of the peak) was the default for years. Its weakness is that the schedule depends on the total number of steps: you cannot extend a finished run or take a cheap intermediate checkpoint that is as good as a fully decayed one.

**Warmup-stable-decay (WSD)** keeps the peak rate for most of training and decays over the last 10–20% of steps. The constant phase can be branched: take any checkpoint, run a short decay, and you get a model nearly as good as a full cosine run of that length. That makes scaling-law sweeps (Chapter 13) and data-mixture experiments much cheaper, and is now common in open model reports.

| Step (of 1,000) | Cosine (peak 3e-4, floor 10%) | WSD (decay over last 20%) |
|---|---|---|
| 50 (warm-up) | 1.50e-04 | 1.50e-04 |
| 500 | 1.88e-04 | 3.00e-04 |
| 800 | 6.16e-05 | 3.00e-04 |
| 900 | 3.81e-05 | 1.50e-04 |
| 999 | 3.00e-05 | 1.50e-06 |

WSD runs show a higher loss than cosine runs during the stable phase, then catch up and usually overtake during the decay. Comparing two runs mid-training with different schedules is therefore meaningless: compare at matched decayed checkpoints.

## 7.3 Batch size, gradient noise and the critical batch size

A mini-batch gradient is the true gradient plus noise whose variance falls as 1/B. Doubling the batch halves the noise. While noise dominates, doubling B lets you nearly double the learning rate and halve the number of steps: **perfect scaling**. Once the noise is small compared with the gradient itself, larger batches only waste compute.

McCandlish et al. (2018) made this quantitative. The **simple noise scale**

B_simple = tr(Σ) / |G|²

(total per-example gradient variance divided by the squared norm of the mean gradient) predicts the **critical batch size**: below it, more batch saves steps almost one for one; above it, returns diminish.

```python
G = per_example_grads(w, all_indices)               # (N, D)
b_simple = G.var(0).sum() / G.mean(0).pow(2).sum()
# logistic regression, 20 features: 35.6 at initialisation, 145,119 after 300 full-batch steps
```

The noise scale **grows during training**: early on, every example agrees on the direction of improvement; near a minimum, the mean gradient shrinks while per-example disagreement remains. That is why LLM runs use batch-size warm-up (start smaller, grow to millions of tokens) and why the critical batch size grows as the loss falls. It also gives the scaling-law version: larger, better-trained models tolerate larger batches.

Rules of thumb:

- **SGD**: scale learning rate linearly with batch size (Goyal et al., 2017), with warm-up.
- **Adam**: closer to square-root scaling, and the effect saturates. Always re-tune the learning rate when you change the batch size by more than about 2×.
- **Tokens, not sequences**: LLM batch sizes are quoted in tokens (for example 4M tokens = 1,024 sequences of 4,096).

## 7.4 Beyond Adam: Adafactor, Lion, Shampoo, SOAP and Muon

**Adafactor** stores the second moment of a matrix as a row vector and a column vector (their outer product approximates it), so the optimiser state is O(m + n) instead of O(mn). Used for T5 and PaLM, useful when memory is tight.

**Lion** (found by program search) keeps only momentum and updates by its **sign**: every coordinate moves by exactly ±η. Half of Adam's state, needs a smaller learning rate (3–10×) and larger weight decay.

**Shampoo and SOAP** precondition each weight matrix with left and right covariance matrices of its gradient, an approximation to second-order methods. They take fewer steps than Adam for the same loss but cost matrix roots and more memory; SOAP runs Adam in Shampoo's eigenbasis and is simpler to tune.

**Muon** (Jordan et al., 2024; scaled up in Moonlight, 2025, and by later open models) treats each hidden weight matrix as a whole. It takes the momentum matrix and replaces it with the nearest orthogonal matrix, so every direction in the update has the same size. Exact orthogonalisation needs an SVD; Muon uses a few Newton–Schulz iterations, which only need matrix multiplications and run well in bf16:

```python
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315          # tuned quintic coefficients
    X = G / (G.norm() + 1e-7)
    transpose = X.shape[0] > X.shape[1]
    if transpose: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transpose else X
# a 64x32 "gradient" with singular values spread 1,889:1 comes out with all of them in 0.17-1.13
```

The intuition: gradient matrices of transformers are dominated by a few directions. Adam follows them; Muon boosts the rare directions to the same size, which explores more of the space per step. Muon applies only to 2-D hidden matrices; embeddings, the output head and norm parameters still use AdamW. Reported gains are of order 1.3–2× fewer tokens for the same loss at small and medium scale; at large scale it needs care with update size (matching Adam's RMS) and with attention logits growing (one open model added a "QK-clip" for that). Treat claimed speed-ups as hypotheses to reproduce on your own setup with tuned baselines: many optimiser papers compare against an under-tuned Adam.

## 7.5 Initialisation, residual scaling and µP

**The variance argument** (Chapter 5): a linear layer with fan-in n and weights of variance σ² multiplies the variance of its input by nσ². σ² = 1/n (Xavier/LeCun, or 2/n for ReLU, He) keeps activations from exploding or vanishing through depth.

**Residual streams** add each block's output to a running sum. With L blocks of unit-variance outputs, the stream's variance grows like L. GPT-2 scaled the output projection of each block by 1/√(2L); many recipes initialise those projections smaller or to zero. Pre-norm (normalise the input of each block, not its output) keeps gradients well behaved at depth and is the standard.

**µP.** In the standard parametrisation (SP), the best learning rate drifts as the model widens: a rate tuned at width 256 makes width 8,192 diverge, so every size needs its own sweep. The **maximal update parametrisation** (Yang et al., 2022) changes the initialisation and the per-layer learning rates so that each layer's update has the same effect on its output at every width. For Adam the main rules are: hidden-matrix learning rates scale as 1/width, the output layer is initialised with variance 1/width² (not 1/width) and attention logits are scaled by 1/d instead of 1/√d.

The check script shows the core problem. One Adam step on a three-layer MLP, same learning rate, three widths:

| Width | Mean change in output, standard | Mean change in output, µP |
|---|---|---|
| 64 | 0.110 | 0.029 |
| 256 | 0.802 | 0.031 |
| 1,024 | 5.639 | 0.030 |

Under SP, the same step moves the output 50× more at width 1,024 than at 64: a learning rate that is safe for the small model blows up the large one. Under µP the effect is constant, so the best learning rate found on a small proxy model transfers. This is **µTransfer**: tune on a 40M-parameter proxy, train the 10B model once. Several labs use µP or close variants; others fit learning-rate scaling laws instead (Chapter 13). Either way, "we tuned hyperparameters at small scale and they transferred" is a claim to check, not assume.

## 7.6 Numerical precision

| Format | Exponent / mantissa bits | Largest value | Notes |
|---|---|---|---|
| fp32 | 8 / 23 | 3.4e38 | Master weights, optimiser states, reductions |
| bf16 | 8 / 7 | 3.39e38 | fp32's range, coarse precision (ε = 0.0078); default for training |
| fp16 | 5 / 10 | 65,504 | Finer precision, tiny range: needs loss scaling |
| fp8 e4m3 | 4 / 3 | 448 | Forward activations and weights |
| fp8 e5m2 | 5 / 2 | 57,344 | Gradients (need range more than precision) |
| MXFP8 / MXFP4 / NVFP4 | 8- or 4-bit elements + a shared scale per block of 16–32 values | – | Block scaling; native on recent GPUs |

**Why fp32 master weights.** bf16 has 8 bits of precision. Adding a small update to a weight of 1.0 loses it entirely:

```python
torch.tensor(1.0).bfloat16() + torch.tensor(1e-3).bfloat16()   # -> 1.0 (the update vanished)
torch.tensor(1.0) + 1e-3                                        # -> 1.001
```

So mixed-precision training computes in bf16 but keeps an fp32 copy of the weights and accumulates updates there. fp16's problem is the opposite: small gradients underflow to zero, hence dynamic **loss scaling** (multiply the loss by a large factor, unscale before the update, skip steps that overflow). bf16 removed the need for it, which is why bf16 won.

**fp8 needs scaling.** e4m3 represents 300 as 288 and anything below about 0.002 as zero:

```python
small = torch.tensor([1e-4, 3e-4, 5e-4])
small.to(torch.float8_e4m3fn)                     # -> all zeros
scale = 448 / small.abs().max()                   # per-tensor scale into the representable range
(small * scale).to(torch.float8_e4m3fn).float() / scale   # -> 9.8e-05, 2.86e-04, 5e-04
```

Production fp8 training (DeepSeek-V3 was the first widely documented frontier-scale example) uses fine-grained scales (per tile or block), keeps sensitive parts in higher precision (embeddings, the output head, normalisation, attention softmax, optimiser states, master weights) and accumulates matrix products in higher precision. MX formats build the block scale into the format itself. The payoff is roughly 2× matmul throughput and less memory; the cost is engineering and the risk of silent quality loss that only appears late in a run. Always validate a low-precision recipe against a bf16 baseline on the same data before trusting it.

## 7.7 Stability: spikes, logits and the edge of stability

Large runs fail in a few recognisable ways.

**Loss spikes.** The loss jumps by a large amount, then either recovers or diverges. Common causes: a bad batch (corrupted or extremely repetitive data), attention logits growing until softmax saturates, the output logits drifting, or Adam's second moment going stale for a parameter that suddenly gets a large gradient. Standard responses, cheapest first: skip the offending batches and restart from the last good checkpoint; lower the learning rate; add the architectural fixes below.

**Gradient clipping** rescales the whole gradient vector when its norm exceeds a threshold (1.0 is standard):

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
# gradient pieces [3,0,0] and [0,4,0,0]: norm 5.0 before, 1.0 after (direction unchanged)
```

Log the pre-clip norm: a run where clipping fires on most steps has a learning rate or stability problem that clipping is hiding.

**z-loss** (PaLM) adds a small penalty on log Z, the log of the softmax normaliser, keeping output logits from drifting upwards:

```python
log_z = torch.logsumexp(logits, dim=-1)       # [12.17, 2.46] for two example rows
z_loss = 1e-4 * log_z.pow(2).mean()           # 0.0077, added to the cross-entropy
```

**QK-norm** normalises queries and keys (RMSNorm on each) before their dot product, so attention logits cannot grow without bound. **Logit soft-capping** (cap · tanh(x / cap)) bounds logits smoothly (used by Gemma 2). Both trade a little flexibility for much better stability at scale.

**Edge of stability.** Cohen et al. (2021) showed that full-batch gradient descent raises the sharpness (largest Hessian eigenvalue) of the loss until it reaches about 2/η, then oscillates there while the loss keeps falling non-monotonically. Classical theory says such a learning rate should diverge. Adam shows an analogous behaviour on its preconditioned sharpness. The practical lesson: the learning rate partly determines which regions of the loss landscape training can stay in, so "a larger learning rate finds flatter minima" has a mechanism, and instability at too high a rate is a sharpness problem, not bad luck.

## 7.8 Reading training curves

| What you see | Likely cause | First thing to try |
|---|---|---|
| Loss flat from step 0 | Learning rate far too small, frozen parameters, bug in labels or masking | Overfit one batch; check that gradients are non-zero everywhere |
| Loss NaN in the first few hundred steps | No or too short warm-up, learning rate too high, fp16 overflow | Longer warm-up, lower rate, bf16, check the data for empty or huge examples |
| Spikes that recover | Bad batches, Adam state going stale | Log the batch IDs at spikes; skip data; lower β₂ |
| Spikes that diverge late in training | Logit growth in attention or output | QK-norm, z-loss, lower rate; restart from before the spike |
| Train loss falls, validation loss rises | Overfitting or repeated data (epochs on a small corpus) | More data, deduplication, earlier stop |
| Loss drops sharply at the end | The schedule decaying (expected for WSD and cosine) | Compare runs only at matched points |
| Two seeds differ more than your ablation effect | The effect is noise | Report seed variance (Chapter 34) |
| Gradient norm rising steadily | Growing logits or weights; weight decay off | Check weight-decay groups; watch per-layer norms |

What to log on every run: loss (train and held-out), learning rate, gradient norm before clipping, per-layer weight and update norms, max attention logit, tokens seen, throughput and MFU (Chapter 17). Most stability problems are visible in these signals thousands of steps before the loss shows them.

## Pitfalls

- Comparing optimisers or architectures at one learning rate. Sweep the rate for each, or the comparison only shows which one likes your default.
- Applying weight decay to norms, biases and embeddings by accident (the default parameter group in many scripts).
- Changing the batch size without re-tuning the learning rate.
- Comparing a cosine run and a WSD run before both have decayed.
- Accumulating updates in bf16 without fp32 master weights.
- Trusting an fp8 or new-optimiser result without a matched baseline at the same token budget.
- Reading one seed as a result.

## Interview questions

**1. Write the AdamW update. Why is weight decay "decoupled"?** See 7.1. With Adam, an L2 term in the gradient is divided by √v, so parameters with large gradients get almost no regularisation; AdamW subtracts lr·wd·p directly, independent of the gradient statistics.

**2. Why do transformers need warm-up?** Adam's first steps have a poorly estimated second moment, so every coordinate moves by about the full learning rate in a random-looking direction; with large rates this throws the model into a bad region or blows up attention logits. Warm-up keeps the early steps small while the statistics settle.

**3. What is the WSD schedule and why do teams like it?** Warm-up, a long constant phase, then a short decay. Any checkpoint in the constant phase can be decayed into a good model, so you can extend runs, run cheap scaling-law and data experiments from branches, and avoid fixing the total length in advance.

**4. What is the critical batch size?** The batch size beyond which larger batches give diminishing reductions in the number of steps. It is roughly the gradient noise scale tr(Σ)/|G|², grows as the loss falls, and is larger for bigger, better-trained models.

**5. How does Muon differ from Adam?** Adam normalises each coordinate separately. Muon orthogonalises the whole momentum matrix of a hidden layer (via Newton–Schulz), equalising the size of the update in all directions. It applies to 2-D hidden matrices only; embeddings and heads stay on AdamW.

**6. What problem does µP solve? State its main rules for Adam.** Hyperparameters, especially the learning rate, stop transferring across widths under the standard parametrisation. µP scales hidden learning rates as 1/width, initialises the output layer with variance 1/width² and scales attention by 1/d, so the per-step change in each layer's output is width-independent and tuned settings transfer from small proxies.

**7. Why bf16 rather than fp16 for training? Why keep fp32 master weights?** bf16 has fp32's exponent range, so gradients do not underflow and no loss scaling is needed. Its 8-bit precision cannot represent small updates to weights of order 1 (1 + 1e-3 rounds to 1), so updates are accumulated in an fp32 copy.

**8. What makes fp8 training hard?** Tiny dynamic range (e4m3 tops out at 448 and flushes values below about 0.002), so tensors need fine-grained scaling factors; some operations and states must stay in higher precision; errors can accumulate silently. Needs a careful recipe and a bf16 baseline.

**9. Your loss spiked at step 60k and did not recover. Walk through what you do.** Restart from a checkpoint before the spike; inspect the batches at the spike; check logged gradient norms, attention max logits and output log Z for the steps before; skip suspicious data and lower the rate; if logits were growing, add QK-norm or z-loss in future runs. Record it, because spikes recur.

**10. What are z-loss and QK-norm for?** z-loss penalises log² of the softmax normaliser to stop output logits drifting; QK-norm normalises queries and keys so attention logits stay bounded. Both prevent the logit growth that causes late-training instability.

**11. What is the edge of stability?** Full-batch GD increases sharpness until it hovers at 2/η, where classical analysis predicts divergence, while the loss still falls non-monotonically. It shows that the learning rate shapes which minima are reachable.

**12. A paper claims a new optimiser trains 2× faster. What do you check?** That the Adam baseline was tuned as hard (learning rate, β₂, warm-up, weight decay) at the same batch size and schedule; that the comparison is at matched tokens and final decayed loss, across several scales and seeds; and that the gain holds on downstream evaluation, not only on loss.

## Practice

1. Implement AdamW, Lion and a Muon step from scratch for a small transformer (Chapter 10's mini-GPT); check AdamW against `torch.optim.AdamW`.
2. Train the mini-GPT with cosine and WSD schedules for the same number of tokens; decay a WSD branch from the middle of the stable phase and compare its loss with a full cosine run of that length.
3. Measure the gradient noise scale at several points of training and plot it against the loss.
4. **Research exercise.** Sweep the learning rate (7 values, 3 seeds each) for widths 128, 256 and 512, under the standard parametrisation and under µP. Plot loss against learning rate per width. Does the optimum move under SP and stay under µP? Report seed variance as error bars.
5. Train in bf16 with and without fp32 master weights; plot the gap.

## Read next

- Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (2019).
- McCandlish et al., "An Empirical Model of Large-Batch Training" (2018).
- Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2022), and the companion µP practitioner notes.
- Keller Jordan's Muon write-up (2024) and Liu et al., "Muon is Scalable for LLM Training" (Moonlight, 2025).
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022), and the DeepSeek-V3 technical report (2024), section on fp8 training.
- Cohen et al., "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability" (2021).
- Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities" (2023): reproduces spikes cheaply and tests the fixes.
- Links to these papers are in the Key concepts section at the top of this chapter.

## Done when

You can write AdamW with bias correction from memory and explain each hyperparameter. You can explain warm-up, cosine and WSD schedules, and why WSD makes experiments cheaper. You can define the gradient noise scale and explain why the critical batch size grows during training.

You can explain what Muon, Lion and Adafactor change, and when you would try them. You can state the problem µP solves and its main rules. You can explain bf16, fp16 and fp8, master weights and fp8 scaling. You can diagnose a loss spike from logged signals and list the standard fixes.

---

[← 6 · PyTorch for practitioners](06-pytorch-for-practitioners.md) · [Contents](../README.md#contents) · [8 · Embeddings and sequence models →](08-embeddings-and-sequence-models.md)
