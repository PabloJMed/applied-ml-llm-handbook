# 11 · Architectures beyond the vanilla transformer

Part III · Transformers and architectures. Assumes Chapter 10 (attention, GQA, the KV cache, RoPE). Chapter 17 covers how mixture-of-experts models are trained across many GPUs; Chapter 21 covers how these choices change serving.

## Why it matters

The 2017 transformer is still the backbone, but almost no frontier model uses it unchanged. The open models that set the pace in 2025–2026 are mixtures of experts with compressed or sparse attention, and a growing share mix attention with linear-time recurrent layers. These choices decide what a model costs to train, how long a context it can serve and how fast it generates. A research scientist is expected to know what each change buys, what it costs, and how to evaluate the claim that a new architecture "matches the transformer": which is where most such claims fail.

Numbers in this chapter come from `code/ch11_check.py` (CPU, seconds).

## Key concepts

- **Mixture of experts (MoE)**: replace the feed-forward block with many expert FFNs and a router that sends each token to a few of them. [Switch Transformer](https://arxiv.org/abs/2101.03961), [GShard](https://arxiv.org/abs/2006.16668), [DeepSeekMoE](https://arxiv.org/abs/2401.06066).
- **Load balancing**: auxiliary losses or bias adjustments that stop the router from collapsing onto a few experts. [Auxiliary-loss-free balancing](https://arxiv.org/abs/2408.15664).
- **Multi-head latent attention (MLA)**: cache a small latent vector per token instead of full keys and values. [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [DeepSeek-V3](https://arxiv.org/abs/2412.19437).
- **Linear attention and state-space models (SSMs)**: sequence mixers with a fixed-size recurrent state and linear cost in length. [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060), [Gated DeltaNet](https://arxiv.org/abs/2412.06464).
- **Hybrid models**: mostly linear-time layers with a few full-attention layers. [Jamba](https://arxiv.org/abs/2403.19887).
- **Multi-token prediction (MTP)**: predict several future tokens per position. [Gloeckle et al.](https://arxiv.org/abs/2404.19737).
- **Diffusion language models**: generate by iteratively unmasking tokens in parallel. [LLaDA](https://arxiv.org/abs/2502.09992).

## 11.1 Three costs that drive architecture research

1. **Compute per token.** A dense transformer spends about 2 FLOPs per parameter per token in the forward pass. Bigger models know more but cost proportionally more for every token, in training and inference.
2. **Attention's quadratic cost.** Self-attention compares every token with every earlier one: O(T²) compute in training and prefill. At 128k tokens this dominates.
3. **KV-cache memory.** At generation time each layer stores keys and values for every past token. The cache grows linearly with context and batch size, and decode speed is limited by reading it from memory (Chapter 21).

MoE attacks the first, sparse and linear attention the second, and GQA, MLA and recurrent layers the third. Most current designs combine several.

## 11.2 Mixture of experts

**The idea.** Replace the single FFN in a block with E expert FFNs and a small router. For each token, the router produces a score per expert; the token goes to the top k experts and their outputs are mixed with the (renormalised) router weights. Total parameters grow with E, but compute per token grows only with k.

```python
class MoE(nn.Module):
    def forward(self, x):                          # x: (tokens, d)
        probs = F.softmax(self.router(x), dim=-1)  # (tokens, n_experts)
        topv, topi = probs.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)   # renormalise the chosen gates
        out = torch.zeros_like(x)
        for e in range(self.n):
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if rows.numel():
                out[rows] += topv[rows, slot, None] * self.experts[e](x[rows])
        f = torch.bincount(topi.flatten(), minlength=self.n).float() / topi.numel()
        P = probs.mean(0)
        aux = self.n * (f * P).sum()               # load-balancing loss
        return out, aux
```

**Load balancing.** Without pressure, the router collapses: a few experts get most tokens, train faster, get chosen more. The Switch/GShard auxiliary loss multiplies, for each expert, the fraction of tokens it receives (f) by its mean router probability (P), times E. It equals 1.0 for perfectly uniform routing and grows with imbalance: 2.52 for the skewed load in the check script. It is added to the language-model loss with a small weight (around 0.01). Too large a weight hurts quality, because it forces routing to ignore content.

**Auxiliary-loss-free balancing** (used in DeepSeek-V3) adds a per-expert bias to the router scores *only for choosing* the top k, not for weighting outputs. After each step, overloaded experts' biases go down and underloaded ones go up. In the check script, a router with a built-in preference starts with expert loads between 0.009 and 0.350 and ends, after 200 updates, between 0.123 and 0.127, with no gradient interference in the main loss.

**The modern design space** (DeepSeekMoE onwards, and most large open models since):

- **Fine-grained experts**: many small experts (64–384) rather than 8 large ones, with more of them active (top 6–8). More combinations of specialists per token.
- **Shared experts**: one or two experts every token uses, holding common knowledge so routed experts can specialise.
- **Sparsity ratio**: total-to-active parameter ratios of 10–30× are common. For one layer with 128 routed experts of width 1,024, one shared expert, top 8 and d_model 4,096, the check script gives 1,623M total and 113M active parameters, less compute than a dense FFN of the usual size (134M).

**What MoE costs.** All experts must sit in memory, so serving needs the total parameter count's memory, spread across GPUs with **expert parallelism** (Chapter 17), and all-to-all communication to send tokens to their experts. Training is less stable (router logits grow; a router z-loss helps), batches must be large enough that each expert sees enough tokens, and fine-tuning can be tricky. Capacity limits (a maximum number of tokens per expert) drop overflow tokens in some implementations, which makes outputs depend on what else is in the batch.

**Why it wins.** At a fixed training compute budget, a well-balanced MoE reaches a lower loss than a dense model with the same active parameters, because it stores more knowledge per FLOP. Scaling-law studies of MoE (Chapter 13) quantify this as an effective-parameter multiplier that shrinks as the dense model grows but stays well above 1 at practical scales.

## 11.3 Attention variants for memory and length

**MQA and GQA** (Chapter 10) share key and value heads across query heads: 8 KV heads instead of 128 shrink the cache 16×.

**Multi-head latent attention (MLA)**, from DeepSeek-V2, goes further. Each token's keys and values are produced from a single low-dimensional latent vector (512 dimensions in DeepSeek-V3), and only that latent is cached; per-head keys and values are reconstructed on the fly, and the up-projections can be folded into the query and output projections. RoPE does not commute with that folding, so a small separate set of "decoupled" rotary dimensions (64) carries position. Per token per layer in bf16:

| Scheme (128 heads of 128 dims) | Cache bytes per token per layer |
|---|---|
| Multi-head attention | 65,536 |
| GQA with 8 KV heads | 4,096 |
| MLA, latent 512 + RoPE 64 | 1,152 |

For a 61-layer model at a 128k-token context that is 524 GB per sequence with full MHA against 9.2 GB with MLA: the difference between impossible and routine.

**Sliding-window and local/global attention.** Each token attends only to the last W tokens (for example 4k) in most layers, with full attention every few layers. Cache and compute for the local layers are bounded by W. Mistral 7B used sliding windows; Gemma 2 and 3 interleave local and global layers.

**Sparse attention.** Each query attends to a selected subset of earlier tokens, chosen by a cheap scoring step: compressed blocks plus top-k selected blocks plus a local window (Native Sparse Attention, 2025), or a lightweight indexer choosing the top-k tokens (DeepSeek Sparse Attention, 2025). The goal is full-attention quality at a fraction of long-context cost, with kernels designed for the hardware from the start.

## 11.4 Linear attention and state-space models

**Linear attention.** Softmax attention needs all past keys. Replace exp(q·k) with a product of feature maps φ(q)·φ(k) and the sum over the past can be accumulated in a d × d state:

S_t = S_{t−1} + φ(k_t) v_tᵀ,  output_t = φ(q_t)ᵀ S_t / φ(q_t)ᵀ z_t

That is an RNN with a matrix-valued state: constant memory per token at inference, and a parallel form for training. The check script confirms the two forms give identical outputs.

**The problem: a fixed state is a fixed-size memory.** The additive update only ever adds; old associations blur together. In the check script, writing key 0 → value 2, then key 1 → value 3, then key 0 → value 1, and querying key 0, returns a mixture of the old and new values ([0, 3, 3, 0]).

**The fixes, in the order they appeared:**

- **Gates / decay.** Multiply the state by a forget factor each step (RetNet, GLA, Mamba's selective decay). Old information fades instead of piling up.
- **Selective state spaces (Mamba).** A structured SSM whose transition and input matrices depend on the current token, so the model chooses what to keep. Mamba-2 showed that a restricted form is a kind of linear attention with a decaying mask ("state-space duality"), which allows matmul-based training kernels.
- **The delta rule (DeltaNet, Gated DeltaNet).** Before writing value v for key k, subtract what the state currently returns for k: S ← S + β k (v − kᵀS)ᵀ. It overwrites rather than accumulates; with a gate it can also forget. In the same test it returns exactly the newest value ([0, 3, 0, 0]).
- **RWKV and xLSTM** are recurrent architectures in the same family with their own gating designs.

**Where they stand.** Recurrent layers match transformers on perplexity at moderate scale and are far cheaper at long context: the recurrent state below is 0.03 GB regardless of length, against an attention KV cache that reaches 137 GB at a million tokens for an 8B-shaped model. Their weakness is **exact recall from long context**: copying, retrieving a specific fact from far back, multi-hop lookups. Benchmarks for associative recall and needle retrieval separate them clearly from attention. That weakness is why the field converged on hybrids.

## 11.5 Hybrid models

Keep a minority of full-attention layers for precise retrieval and make the rest linear-time:

- **Jamba** (2024): Mamba layers with one attention layer in eight, plus MoE.
- **Nemotron-H** (2025): mostly Mamba-2 layers with a few attention layers, aimed at inference throughput.
- **Qwen3-Next** (2025): Gated DeltaNet layers interleaved 3:1 with gated full attention, plus a high-sparsity MoE.
- Others interleave "lightning" or other linear attention with softmax attention at similar ratios.

The design questions are the ratio (1:3 to 1:8 is typical), where the attention layers go (not all at the start), whether attention layers use positional encodings at all, and how to evaluate. Perplexity alone hides the recall weakness; evaluate on long-context retrieval (RULER, needle-in-a-haystack variants), in-context learning and real long-document tasks, at matched training tokens.

## 11.6 Multi-token prediction

Standard training predicts only the next token. **Multi-token prediction** adds heads (or small extra transformer modules, as in DeepSeek-V3) that predict tokens t+2, t+3, ... from the same trunk. Two motivations:

- **Training signal.** Planning a few tokens ahead may improve representations, especially for code; reported gains are mixed and scale-dependent.
- **Faster inference.** The extra heads are a built-in draft model for **speculative decoding** (Chapter 21): propose several tokens, verify them in one forward pass. This is often the bigger practical win.

## 11.7 Diffusion language models

Autoregressive models generate one token at a time, left to right. **Masked diffusion models** start from a fully masked sequence and, over a number of steps, predict all masked positions in parallel, keep the most confident predictions and re-mask the rest.

Training is masked-token prediction with a random mask rate t per sequence and a 1/t weight, which makes the loss a bound on the log-likelihood:

```python
t = torch.rand(batch, 1).clamp(min=0.05)          # mask rate per sequence
masked = torch.rand(batch, L) < t
inp = tokens.masked_fill(masked, MASK)
ce = F.cross_entropy(model(inp).transpose(1, 2), tokens, reduction="none")
loss = ((ce * masked) / t).sum() / (batch * L)
```

LLaDA (2025) trained an 8B diffusion LM that was competitive with autoregressive models of similar size on many benchmarks, and commercial diffusion models for code have shown very high generation speeds. Open questions: they need many denoising steps for quality, KV caching does not apply directly, and variable-length generation and reasoning-style long outputs are awkward. Worth knowing; not yet the default.

## 11.8 Evaluating a new architecture

Most "X beats the transformer" claims dissolve under these checks:

| Check | Why |
|---|---|
| Matched training tokens and data | More or better data explains most wins |
| A strong, tuned transformer baseline (modern recipe: RMSNorm, SwiGLU, RoPE, GQA, tuned learning rate) | The 2017 baseline is easy to beat |
| Several scales, with a scaling-law fit (Chapter 13) | Advantages at 125M often vanish at 1B+ |
| Recall and long-context tasks, not only perplexity | Recurrent models hide their weakness in perplexity |
| Wall-clock and memory on real hardware | Linear in theory is not fast without kernels |
| Seeds and confidence intervals (Chapter 23) | Small differences are often noise |

## Pitfalls

- Quoting an MoE's total parameters as if it were a dense model of that size, or its active parameters when discussing memory.
- Forgetting that every expert must be loaded for serving, even though each token uses few.
- Concluding that a recurrent model "handles 1M tokens" because it runs at that length; check whether it retrieves from that length.
- Comparing architectures at one learning rate (Chapter 7).
- Treating the 1/t weighting in masked diffusion as optional: without it the loss is not a likelihood bound.

## Interview questions

**1. How does a top-k MoE layer work, and what does it change in compute and memory?** A router scores experts per token; the token goes through the top k experts and their outputs are mixed. Compute per token scales with k, memory with the total number of experts; training and serving need expert parallelism and all-to-all communication.

**2. Why do MoE models need load balancing? Give two approaches.** Routers collapse onto a few experts. The Switch/GShard auxiliary loss E·Σ f_e P_e pushes towards uniform load; auxiliary-loss-free balancing adjusts a per-expert selection bias up or down by load, avoiding gradient interference.

**3. What are fine-grained and shared experts?** Many small experts with more of them active, giving more combinations; and one or more always-on experts holding common knowledge so routed ones specialise. Introduced in DeepSeekMoE and now standard.

**4. Explain MLA. Why does it need decoupled RoPE?** Keys and values are reconstructed from a cached low-rank latent per token, cutting the cache by tens of times. The up-projections are absorbed into the query and output matrices at inference, which only works if no position-dependent rotation sits between them, so position goes through a small extra set of rotary dimensions.

**5. Show that causal linear attention is an RNN.** With φ(q)·φ(k) instead of exp(q·k), the sums Σ φ(k_s) v_sᵀ and Σ φ(k_s) can be accumulated step by step; each output needs only the current state.

**6. What is the main weakness of SSMs and linear attention? How do hybrids address it?** A fixed-size state cannot store everything, so precise recall from long context degrades. Hybrids keep a few full-attention layers for retrieval and make the rest linear-time.

**7. What does the delta rule add to linear attention?** It subtracts the state's current prediction for a key before writing the new value, so it overwrites associations instead of summing them; combined with a gate it can forget.

**8. What is multi-token prediction good for?** Extra training signal from predicting several tokens ahead, and built-in draft heads for speculative decoding.

**9. A paper says its linear-time model beats a transformer at 350M parameters. What do you ask?** Same data and tokens? Modern, tuned baseline? Does it hold at larger scale? Recall and long-context evaluations? Real throughput with kernels? Seeds?

**10. When would you choose a dense model over an MoE?** Small deployments where memory, not compute, is the constraint; on-device; heavy fine-tuning by teams without MoE expertise; very low batch sizes where experts are under-used.

## Practice

1. Add the MoE layer above to the mini-GPT of Chapter 10 (8 experts, top 2). Train with and without the auxiliary loss and plot expert loads over time.
2. Implement auxiliary-loss-free balancing and compare its final loss and load balance with the auxiliary loss at three loss weights.
3. Implement chunked causal linear attention (parallel within chunks, recurrent between them) and check it against the recurrent form.
4. **Research exercise.** Build a synthetic associative-recall task (sequences of key–value pairs, then a query key). Compare softmax attention, additive linear attention and the delta rule at sequence lengths 64, 256 and 1,024, three seeds each. Plot accuracy against length.
5. Compute the KV cache per sequence for a model you use, under MHA, GQA and MLA, at 8k and 128k tokens.

## Read next

- Fedus, Zoph and Shazeer, "Switch Transformers" (2021); Dai et al., "DeepSeekMoE" (2024).
- DeepSeek-AI, "DeepSeek-V2" (2024) for MLA and "DeepSeek-V3 Technical Report" (2024) for auxiliary-loss-free balancing, MTP and fp8.
- Gu and Dao, "Mamba" (2023); Dao and Gu, "Transformers are SSMs" (Mamba-2, 2024).
- Yang et al., "Gated Delta Networks" (2024).
- Lieber et al., "Jamba" (2024).
- Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models" (2023): why recall is the right test.
- Nie et al., "Large Language Diffusion Models" (LLaDA, 2025).
- Links to these papers are in the Key concepts section at the top of this chapter.

## Done when

You can implement a top-k MoE layer with a load-balancing loss and explain auxiliary-loss-free balancing. You can compute total and active parameters and explain what MoE costs in serving. You can explain MLA and compute KV-cache sizes for MHA, GQA and MLA.

You can show that linear attention is an RNN, explain the recall weakness of fixed-size states and what gates and the delta rule fix. You can describe a hybrid design and how you would evaluate it. You can explain multi-token prediction and masked diffusion LMs, and list the checks a new-architecture claim must pass.
