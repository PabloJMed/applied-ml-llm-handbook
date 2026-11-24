# 12 · Model families and scaling

Part III · Transformers and architectures. Assumes Chapter 10 (the transformer block). Chapter 11 goes deeper on architecture research, Chapter 13 on scaling laws, Chapter 14 on multimodal models.

## Why it matters

"Which model would you use and why?" comes up in every LLM interview and every design review. A senior answer does not name the newest frontier model. It names the right *family* for the task: an encoder for classification at 20k requests per second, a decoder for generation, a bi-encoder for retrieval, a vision-language model for screenshots and scanned documents. It then sizes the model with scaling arithmetic and justifies the cost. This chapter gives you that map; Chapter 10 covers the transformer block itself, and Chapter 15 covers how these models are trained.

The chapter is organised by the questions you face when choosing a model. Which family fits the task (12.1)? How do you get good sentence embeddings for search and clustering (12.2)? Sections 12.3 to 12.5 are short pointers that give the facts you need to read a model card (what "total" and "active" parameters mean, what CLIP is, what Chinchilla says) and send you to the chapters that teach each topic properly. Section 12.6 turns this into a decision for a concrete workload, with FLOPs, memory bandwidth, price and latency. Section 12.7 lists where to go deeper, 12.8 explains how to read a model card, and 12.9 places the 2025–26 open-weight families on one map.

The aim is fluency in the vocabulary and the arithmetic, so that you can place any new model quickly: what family it is, roughly what it costs to run, and what it is likely to be good at. Interviewers test exactly that. Given "a 235B-A22B MoE with 128k context under Apache 2.0", can you say in a minute how many GPUs it needs, how fast it decodes compared with a dense 22B model, and whether you may ship a distilled version? Most candidates can name models; fewer can do this arithmetic.

## Key concepts

- **Encoder-only (BERT family)**: bidirectional models pre-trained by masked language modelling; best for classification, extraction, embeddings and reranking. [BERT](https://arxiv.org/abs/1810.04805), [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/).
- **Decoder-only (GPT family)**: causal models trained to predict the next token; nearly all LLMs. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- **Encoder–decoder (T5 family)**: an encoder reads, a decoder writes; every task framed as text-to-text. [T5](https://arxiv.org/abs/1910.10683).
- **Sentence transformers**: encoders fine-tuned so pooled vectors work as sentence embeddings. [Sentence-BERT](https://arxiv.org/abs/1908.10084).
- **Mixture of experts (MoE)**: a router sends each token to a few expert networks; more parameters at the same compute. [Switch Transformers](https://arxiv.org/abs/2101.03961), [Mixtral](https://arxiv.org/abs/2401.04088).
- **CLIP and vision-language models**: images and text in a shared space; LLMs that accept images. [CLIP](https://arxiv.org/abs/2103.00020), [LLaVA](https://arxiv.org/abs/2304.08485).
- **Scaling laws and Chinchilla**: loss falls predictably with parameters, data and compute; compute-optimal training uses about 20 tokens per parameter. [Kaplan et al.](https://arxiv.org/abs/2001.08361), [Chinchilla](https://arxiv.org/abs/2203.15556), [Besiroglu et al. replication](https://arxiv.org/abs/2404.10102).
- **Emergent abilities, debated**: [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004)
- **Open-weight landscape**: the families you will be asked to compare (Llama, Qwen, DeepSeek, Gemma, Mistral, gpt-oss, Kimi), described by dense or MoE, active parameters, context, licence and reasoning variants. [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [Qwen3](https://arxiv.org/abs/2505.09388), [Gemma 3](https://arxiv.org/abs/2503.19786).

## 12.1 Three families

Every transformer model belongs to one of three families, and the difference between them is surprisingly small: which tokens each position is allowed to attend to, and what the model is trained to predict. That single choice decides whether a model is best at understanding text, generating it, or transforming one text into another.

![The architecture family is mostly a choice of attention mask plus a training objective](images/12-diagram-1.png)

*The architecture family is mostly a choice of attention mask plus a training objective.*

**Encoder-only: BERT and descendants** (RoBERTa, DeBERTa, XLM-R, ModernBERT). Pre-trained with *masked language modelling*: hide about 15% of tokens and predict them from both sides. BERT also used next-sentence prediction, which RoBERTa showed was unnecessary; RoBERTa's lesson was that more data, bigger batches and longer training mattered more than the objective tweaks. The output is a contextual vector per token, plus a `[CLS]` vector for the whole input. Best for classification, NER, extraction, cross-encoder reranking and embeddings. They are small (100M–400M parameters), fast, and still often the best choice for a high-volume, fixed-label task. XLM-R and multilingual BERT cover about 100 languages; for a specific language or dialect, a language-specific encoder (CamemBERT for French, AraBERT or MARBERT for Arabic, and similar models for most large languages) is often worth testing against them.

**Decoder-only: GPT and descendants** (Llama, Mistral, Qwen, Gemma, Claude, GPT-4-class models). Pre-trained to predict the next token with a causal mask. Generation, few-shot learning and instruction following come from this one objective at scale, which is why nearly all LLMs are decoders. Weakness: they only read left to right, so as raw embedders they are worse than encoders unless adapted (12.2 explains how they are adapted, and why the best embedders are now decoder-based).

**Encoder-decoder: T5, BART, mT5, Flan-T5.** The encoder reads the input bidirectionally; the decoder generates with causal self-attention plus cross-attention to the encoder. T5 framed every task as text-to-text ("translate English to German: …", "sst2 sentence: …") and used span corruption as the objective. Strong for translation and summarisation at small sizes; less common now at the frontier (Chapter 10, Q12, discusses why and where the gap is smaller than it looks).

**What each objective teaches.** Masked language modelling gives every position information from both sides, so its representations are excellent for understanding a fixed text, but the model never learns to produce text left to right. In BERT, 15% of tokens are selected; of those, 80% are replaced by `[MASK]`, 10% by a random token and 10% left unchanged, so the model cannot learn that only `[MASK]` positions matter (the mask token never appears at fine-tuning time). Next-token prediction uses only the left context, which makes every position a training example (no 85% of positions wasted) and makes generation the native operation. Span corruption (T5) hides whole spans and asks the decoder to write them out, a middle ground suited to input-to-output tasks.

## 12.2 Sentence transformers

Search, clustering, deduplication and retrieval-augmented generation all need one vector per text, with similar texts close together. A pretrained encoder gives one vector per token, and the obvious ways of combining them work poorly. Sentence transformers fine-tune the encoder so that a single pooled vector captures the meaning of the whole text, which is what made dense retrieval practical.

A vanilla BERT `[CLS]` vector is a poor sentence embedding. Sentence-BERT (Reimers and Gurevych, 2019) fixed this by fine-tuning a siamese encoder so that cosine similarity of pooled outputs matches semantic similarity, later with contrastive losses on large pair datasets (E5, GTE, BGE and the multilingual variants). Pooling is usually the masked mean:

```python
def mean_pool(h, mask):              # h: (B, T, d) last hidden states, mask: (B, T)
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
```

**Why `[CLS]` is a poor sentence vector out of the box.** Pretraining never asked the `[CLS]` position to summarise meaning in a way that makes cosine similarity meaningful (next-sentence prediction was a weak signal, and RoBERTa dropped it). Raw BERT representations are also strongly anisotropic: all sentences point in similar directions, so cosines cluster near the top of the range and say little. Mean pooling over tokens is more robust, and contrastive fine-tuning (Chapter 29) is what turns any of them into a usable similarity space.

Bi-encoder (embed separately, compare with a dot product, indexable) vs cross-encoder (feed both texts together, much more accurate, not indexable) is the key retrieval trade-off; Chapters 29–30 go deep.

### Decoder LLMs as embedders

The top of public embedding leaderboards is now held by models built on decoder backbones: E5-Mistral-7B, NV-Embed, gte-Qwen and the Qwen3-Embedding series, among others. This looks paradoxical after the previous section, and the resolution is that "worse as raw embedders" is about the *pooling and attention mask*, not about what the backbone knows. Three adaptations close the gap.

**Pooling: last token, not mean.** Under a causal mask, token t has seen only tokens 1 to t. The first token's hidden state knows almost nothing about the text, so a plain mean over positions is dominated by states that saw only a prefix. The last token (usually an appended end-of-sequence token) is the only position that has read everything, so **last-token pooling** is the default for causal embedders. The alternatives are to keep mean pooling but make attention bidirectional, or to learn a small attention-pooling layer over all positions (NV-Embed's "latent attention").

**Bidirectional conversion.** LLM2Vec (2024) removes the causal mask, briefly trains the model with masked next-token prediction so it learns to use the right context, and then fine-tunes contrastively. NV-Embed also drops the causal mask during contrastive training. Both show that a decoder's knowledge transfers once the mask stops hiding half of the text.

**Instructions on the query side.** Queries are prefixed with a task description, for example "Instruct: Given a web search query, retrieve relevant passages. Query: …", while documents are embedded without one. One model can then produce different embeddings for retrieval, classification or clustering, which is what lets these models top multi-task benchmarks.

**The trade-off, in numbers.** A 7B embedder costs about 2 × 7 × 10⁹ = 1.4 × 10¹⁰ FLOPs per token, about 64 times a 110M BERT-base, and typically outputs 4,096 dimensions rather than 768. One million documents stored as float32 take 16.4 GB at 4,096 dimensions against 3.1 GB at 768 (`code/ch12_check.py`). Use a decoder embedder when retrieval quality on hard, long or multilingual queries is the bottleneck, and pair it with Matryoshka truncation or quantisation (Chapter 29, 29.5–29.6). Keep a small encoder when you re-embed a large corpus often or have a tight latency budget. Section 29.9 covers training these models.

## 12.3 Mixture of experts

Mixture of experts is the main way frontier models grow their parameter count without growing their cost per token. This section gives only the facts you need to read a model card; Chapter 11, section 11.2, teaches the mechanism, the load-balancing losses and the modern design space, with code and checked numbers.

An MoE layer replaces the single feed-forward block with E experts and a router that sends each token to the top k of them. Total parameters grow with E; compute per token grows only with k. A model card therefore quotes two numbers. Mixtral 8×7B has about 47B total and 13B active parameters; DeepSeek-V3 has 671B total and 37B active. **Memory follows the total and speed follows the active count**: all experts must sit in GPU memory, but each token does the FLOPs of the active parameters only. The two serving consequences that matter for choosing a model are covered in 11.2: at small batch, decoding reads only the active experts' weights, while at large batch almost every expert is touched, so the bandwidth advantage shrinks and expert-parallel communication dominates.

**Two gating conventions.** You will see two orders of operations in code, and they are not equivalent. Mixtral takes the top-k router *logits* and applies the softmax over those k only, so the chosen gates always sum to 1. Switch and GShard apply the softmax over *all* experts and then take the top k. With k = 1 (Switch) the single gate is then less than 1, which scales the expert output and keeps a gradient to the router. With k > 1, most implementations renormalise the chosen gates (the version in 11.2). DeepSeek-V3 replaces the softmax with a sigmoid score per expert and normalises over the chosen ones. The choice matters for two reasons. Unnormalised gates change the scale of the layer's output with the router's confidence. And the load-balancing loss needs probabilities over all experts, which only the softmax-first version computes as a by-product.

## 12.4 Multimodal: CLIP and vision-language models

Text-only models cannot read a photo, a screenshot or a chart. The step that made multimodal models possible was learning a shared space in which an image and its description land close together. Chapter 14 covers the full design space (14.2 for vision encoders and the SigLIP alternative, 14.3 to 14.4 for connecting them to an LLM). What follows is the minimum the later chapters assume.

**CLIP** (Radford et al., 2021) trains an image encoder and a text encoder together on hundreds of millions of image-caption pairs with a symmetric contrastive (InfoNCE) loss: in a batch of N pairs, the N true pairs should score higher than the N²−N mismatched ones.

```python
def clip_loss(img, txt, t=0.07):
    img, txt = F.normalize(img, dim=-1), F.normalize(txt, dim=-1)
    logits = img @ txt.T / t                      # (N, N) similarity matrix
    labels = torch.arange(len(img))               # true pairs on the diagonal
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
# aligned toy batch: 0.0001; random pairs: 4.73
```

The code fixes the temperature at 0.07 for clarity. CLIP itself *learns* it: the logit scale 1/t is a trainable parameter (stored as a logarithm), initialised at 1/0.07 ≈ 14.3 and clipped at 100 to keep training stable. The result is a shared embedding space that supports zero-shot classification (compare an image with "a photo of a {label}") and text-to-image search. Chapter 29 derives the same InfoNCE loss for text embedders.

**VLMs** (LLaVA, Qwen-VL, Gemma 3, GPT-4o, Claude, Gemini) connect a vision encoder (often CLIP-like or SigLIP) to an LLM through a projection layer, so image patches become "visual tokens" in the LLM's input. Typical uses are document and chart understanding, screenshot-driven agents, visual question answering and attribute extraction from images and text together; Chapter 14 covers how they are trained, their token budgets and how to evaluate them.

## 12.5 Scaling laws

Training a frontier model costs too much to proceed by trial and error, so labs need to predict the result of a run before paying for it. Scaling laws make that possible: across many orders of magnitude, loss falls as a smooth, predictable function of model size, data and compute. This section gives the two landmark results and the rule of thumb in a form you can compute in an interview; Chapter 13 derives, fits and applies them.

**Kaplan et al. (2020)** found that loss falls as a smooth power law in parameters N, data D and compute C, and concluded that for a fixed compute budget you should mostly grow the model. **Chinchilla (Hoffmann et al., 2022)** redid the analysis with learning-rate schedules matched to each run's length and found that N and D should grow in equal proportion: about **20 training tokens per parameter**. With training compute C ≈ 6·N·D FLOPs (Chapter 10, 10.8, derives the 6) and D = 20N, the budget gives C = 120N², so

$$
N_{\mathrm{opt}} = \sqrt{\dfrac{C}{120}},\qquad D_{\mathrm{opt}} = 20\,N_{\mathrm{opt}}
$$

where C is the training budget in FLOPs, N the number of parameters and D the number of training tokens. For C = 10²³: 10²³/120 = 8.3 × 10²⁰, whose square root is 2.9 × 10¹⁰, so about 29B parameters and 580B tokens.

```python
import math
def chinchilla(C):
    N = math.sqrt(C / (6 * 20)); D = 20 * N
    return N, D
# C = 1e21 -> N = 2.9B params,  D = 58B tokens
# C = 1e23 -> N = 28.9B params, D = 577B tokens
# C = 1e24 -> N = 91.3B params, D = 1,826B tokens

# Their fitted loss L(N, D) = E + A/N^a + B/D^b (E=1.69, A=406.4, B=410.7, a=0.34, b=0.28):
# Chinchilla 70B on 1.4T tokens -> 1.937;  Gopher 280B on 300B tokens -> 1.993
```

Same compute, four times fewer parameters, lower loss: Gopher was undertrained.

### Why Chapter 13 says 14.6B for the same budget

For C = 10²³ FLOPs this chapter gives about 29B parameters, and Chapter 13's worked table (13.3) gives 14.6B. Chapter 13's passage "Three answers for one budget" sets out the same three numbers as the table below. Both are correct calculations; they use different inputs.

| Method | N_opt | D_opt | Tokens per parameter |
|---|---|---|---|
| Rule of thumb, D = 20N | 28.9B | 577B | 20 |
| Besiroglu et al. (2024) re-fit of the parametric loss | 29.4B | 566B | 19 |
| Hoffmann et al.'s third approach (constants above) | 14.6B | 1.14T | 78 |

The Chinchilla paper estimated the optimum three ways. Two agreed on roughly 20 tokens per parameter. The third fitted L(N, D) = E + A/N^α + B/D^β directly, and its published constants imply 78 tokens per parameter at 10²³ FLOPs, drifting upwards with scale. Besiroglu et al. reconstructed the data and showed that this third fit was flawed. Their re-fit (E ≈ 1.82, A ≈ 482, B ≈ 2085, α ≈ 0.35, β ≈ 0.37) gives 29.4B, within 2% of the rule of thumb (`code/ch12_check.py`). Chapter 13 keeps the original constants on purpose, to show how to check a fit by deriving its implications. So **quote about 29B and 20 tokens per parameter**, and treat 14.6B as the worked example of a broken fit. The two fits agree well on loss values (both reproduce Chinchilla beating Gopher) and disagree on the optimum. The optimum depends on the *ratio* of the two error terms, which a slightly wrong fit can move a lot.

**Beyond Chinchilla.** Chinchilla-optimal minimises *training* compute. If the model will serve billions of requests, inference cost dominates, so it pays to train a smaller model on far more data than 20 tokens per parameter. Llama 3 8B was trained on about 15T tokens, close to 1,900 per parameter. Chapter 13 (13.5) does the full inference-aware calculation, and Q6 below does a quick version. The newer axis is **test-time compute**: reasoning models spend more tokens thinking at inference, and accuracy on hard problems keeps rising with that budget (Chapter 13, 13.9, and Chapter 19, 19.8).

**Emergent abilities.** Some abilities seem to appear suddenly with scale. Later work (Schaeffer et al., 2023) argued that much of this is an artefact of discontinuous metrics like exact match; with smooth metrics, progress looks gradual. Q8 gives the full answer, and Chapter 13 (13.8) connects loss to downstream accuracy.

## 12.6 Small vs large: choosing a model

The best model on a leaderboard is rarely the right model for a product. A classifier that handles 20,000 requests per second, an assistant that must answer in under a second, and an offline batch job that labels a million documents each have different constraints. The decision starts with arithmetic: what each candidate costs per request, and whether the quality gap justifies the cost.

**The arithmetic first.** A forward pass costs about 2 FLOPs per parameter per token. Classifying a 64-token text costs about 14 GFLOPs with BERT-base (110M parameters), 1,024 GFLOPs with an 8B LLM and 8,960 GFLOPs with a 70B model:

![Forward-pass compute per 64-token request, 2 × parameters × tokens (the LLMs would also generate at least one output token)](images/12-fig-1.png)

*Forward-pass compute per 64-token request, 2 × parameters × tokens (the LLMs would also generate at least one output token).*

At 20,000 requests per second, the BERT-base option needs about 0.3 PFLOP/s of sustained compute, a handful of GPUs; the 70B option needs about 180 PFLOP/s, hundreds of GPUs. The right question is never "which model is best?" but "which is the smallest model that meets the quality bar on my evaluation set, at my volume and latency?". The usual pattern is to prototype with a large model, measure, then distil or fine-tune a small one for the volume path (Chapters 19 and 41).

| Situation | Usually the right first choice |
|---|---|
| Fixed labels, high volume, tight latency (query intent, content tagging) | Fine-tuned encoder (DeBERTa, XLM-R) or a small LLM distilled from a large one |
| Retrieval and semantic search | Bi-encoder embedding model + cross-encoder reranker |
| Open-ended generation, reasoning, tool use | Large LLM via API |
| Many tasks, few labels, need it next week | Large LLM with good prompts; collect its outputs as labels |
| Same task at scale, after the prototype works | Distil into a small model; keep the large one as teacher and fallback |
| Data must stay in-house or cost per call must be near zero | Open-weight model self-hosted (Chapter 21) |

The path most teams follow: prototype with a large API model, measure on a labelled set, then cut cost with batch APIs, smaller models, cascades (small model first, large model only when unsure) or distillation. A product-categorisation or content-moderation pipeline usually follows exactly this shape.

### Beyond FLOPs: memory bandwidth, price and latency

FLOP counts describe classification and prefill well, because they run as large matrix multiplications. Generation does not. Each decode step reads every weight from GPU memory to produce one token per sequence, so its speed is set by **bytes moved**, not FLOPs (Chapter 10, 10.8). A roofline estimate for an 8B model in bf16 (16 GB of weights) on a GPU with 3.35 TB/s of memory bandwidth goes as follows.

- **Weights only.** One step reads 16 GB, which takes 4.8 ms. With 64 sequences in the batch, each step yields 64 tokens, so the GPU produces about 13,350 tokens/s.
- **With the KV cache.** If each sequence has 2,000 tokens of context, the GQA-8 cache adds 2 × 32 × 8 × 128 × 2 bytes × 2,000 × 64 = 16.8 GB per step. The step time doubles to 9.8 ms, and throughput falls to about 6,530 tokens/s.
- **Price.** At an assumed $2.50 per GPU-hour, that is $0.05 to $0.11 per million output tokens (`code/ch12_check.py`).

Real engines reach perhaps 30–60% of the roofline, so budget two to three times these figures. Three lessons follow:

- **Batching is what makes decoding cheap.** At batch 1 the same GPU produces about 200 tokens/s for the same price per hour, 64 times more per token. Batch size is capped by memory left for the KV cache, so cache size, not FLOPs, often sets your price.
- **Long contexts raise price even at fixed output length**, because the cache is re-read every step and is not shared across sequences.
- **Latency targets cap the batch.** Time to first token (TTFT) is set by prefill FLOPs plus queueing. Time per output token (TPOT) is the step time above, and it grows with batch size and context. An interactive product with a 30 ms TPOT target cannot run the batch sizes an offline job can, so the same model costs several times more per token in the interactive path. Chapter 21 (21.9) does this arithmetic in full.

**When a cascade pays.** A cascade sends every request to a small model and escalates a fraction p to a large one. If the small model costs c_s and the large one c_L per request, the cascade costs c_s + p·c_L, against c_L for the large model alone. With c_s = 0.05·c_L, escalating 20% costs 0.25 of the large-only price, and the cascade stops saving anything at p = 1 − c_s/c_L = 95%. The cost side is rarely the problem; the router's errors are. Every item the small model gets wrong *and is confident about* is never escalated, so the cascade's error rate is at least (1 − p) × the small model's error rate on the items it keeps. If meeting the quality bar forces the threshold so high that p reaches 50%, the saving is only 45%, and escalated requests also pay both models' latency. Before building one, measure on your evaluation set how the small model's accuracy on the items it keeps changes with the threshold. That curve, not the price list, decides whether the cascade is worth it.

## 12.7 Where to go deeper

This chapter is the overview. For the modern mixture-of-experts design space (many fine-grained experts with top-8 routing, shared experts, auxiliary-loss-free balancing) and other architecture changes, see Chapter 11. Most leading open-weight models released since 2024 are MoEs, and the "top-1 or top-2 of 8 experts" design that Switch and Mixtral popularised is the earlier version. For scaling laws in depth (derivations, fitting, inference-aware over-training, data constraints, test-time compute), see Chapter 13. For multimodal models beyond CLIP, see Chapter 14. For how reasoning variants are trained and distilled, see Chapter 19.

## 12.8 Reading a model card

New models arrive weekly, each with a model card or technical report full of benchmark tables. The benchmark tables are the least informative part. Reading the cards systematically, in a fixed order, tells you within minutes what a model is, what it was trained on, and how much its numbers can be trusted.

Every model release comes with a card or report. Read it for these facts, in this order, before you read any benchmark table:

| Item | Why it matters | Where it bites |
|---|---|---|
| Architecture and size: dense or MoE, total and active parameters | Memory comes from total, speed from active | A "30B" MoE may need the memory of a 30B dense model and run like a 3B one |
| Context length, and how it was reached (native vs extended) | Extended contexts often degrade before the advertised limit | Long-document products |
| Tokenizer and vocabulary size | Cost and quality per language (Chapter 9) | Multilingual products |
| Training tokens and data description | Indicates over-training, knowledge cut-off, domain coverage | Freshness, niche domains |
| Post-training: instruction-tuned, reasoning, safety tuning | Decides default behaviour and refusal patterns | Agents, sensitive domains |
| Licence and acceptable-use terms | Whether you can ship it, and fine-tune or distil from it | Every commercial use |
| Evaluation setup: prompts, shots, decoding, contamination checks | Without it, numbers are not comparable (Chapter 23) | Model selection |

Then run the candidate on 100–300 of your own examples. Public benchmarks rank models on average tasks; your evaluation set ranks them on yours, and the two orders often differ.

## 12.9 The open-weight landscape, 2025–26

Interviewers increasingly ask candidates to compare open models: "Why would you pick a Qwen3 MoE over a dense Llama?", "What does 'A3B' mean?", "Can you distil from this model?". The answers come from a handful of attributes that the 12.8 checklist extracts. This section applies the checklist to the families that set the pace from 2025 into 2026. Releases arrive monthly, so learn the pattern rather than the roster; for any newer model, fill in the same columns from its card.

### The map

The table lists the main open-weight families as described in their release cards and reports. Parameter counts are total / active. "Context" is the advertised maximum, which is often reached by extension and should be verified (12.8). Check the current card before quoting a licence: terms change between versions.

| Family (developer, first release) | Sizes | Dense or MoE (total / active) | Context | Licence | Reasoning variants |
|---|---|---|---|---|---|
| Llama 3.1 / 3.3 (Meta, 2024) | 8B, 70B, 405B | Dense | 128k | Llama community licence (custom; extra terms above 700M monthly users) | None separate |
| Llama 4 (Meta, 2025) | Scout, Maverick | MoE: 109B / 17B (16 experts), 400B / 17B (128 experts) | Advertised 10M (Scout), 1M (Maverick) | Llama 4 community licence | None at release |
| Qwen3 (Alibaba, 2025) | 0.6B–32B dense; 30B-A3B, 235B-A22B | Both | 32k native, 128k with YaRN; longer in later updates | Apache 2.0 | Hybrid thinking / non-thinking in one model, later split into separate Instruct and Thinking checkpoints |
| DeepSeek-V3 / R1 (DeepSeek, Dec 2024 / Jan 2025) | One size | MoE: 671B / 37B, MLA attention | 128k | MIT (R1, and V3 from its 2025 update) | R1 trained by RL; later V3.x releases offer a thinking mode in one model; V3.2 adds sparse attention (11.3) |
| Gemma 3 (Google, 2025) | 1B, 4B, 12B, 27B | Dense, 5 local : 1 global attention layers | 128k (32k for 1B) | Gemma terms of use (custom) | None separate |
| Mistral (Mistral AI) | Mixtral 8×7B (2023), Mistral Small 3.x 24B (2025) | Mixtral MoE 47B / 13B; Small dense | 32k (Mixtral), 128k (Small 3.1) | Apache 2.0 for these | Magistral (2025) |
| gpt-oss (OpenAI, Aug 2025) | 20b, 120b | MoE: 21B / 3.6B, 117B / 5.1B | 128k | Apache 2.0 | Reasoning-only, with low / medium / high effort settings |
| Kimi K2 (Moonshot AI, 2025) | One size | MoE: 1T / 32B, 384 experts, top 8, MLA | 128k, extended later | Modified MIT (attribution above a usage threshold) | K2 Thinking (late 2025) |
| GLM-4.5 (Zhipu AI, 2025) | 4.5, 4.5-Air | MoE: 355B / 32B, 106B / 12B | 128k | MIT | Hybrid thinking mode |
| MiniMax-M1 (MiniMax, 2025) | One size | MoE: 456B / 46B, hybrid lightning attention (11.5) | 1M | Apache 2.0 | Reasoning model |
| OLMo 2 (Ai2, 2024–25) | 7B, 13B, 32B | Dense | 4k | Apache 2.0, with training data and code released | Later OLMo releases add thinking variants |
| Phi-4 (Microsoft, Dec 2024) | 14B (plus smaller "mini" models) | Dense | 16k | MIT | Phi-4-reasoning (2025) |

### How to read it

**Above about 100B total parameters, open models are MoEs.** Every family in the table that goes past roughly 100B total, apart from Llama 3.1 405B, is an MoE, and the active share is small: 5.5% for DeepSeek-V3, 3.2% for Kimi K2, 4.4% for gpt-oss-120b, 9.4% for Qwen3-235B-A22B (`code/ch12_check.py`). The naming convention "235B-A22B" or "30B-A3B" states both numbers. Dense models dominate below about 32B, where memory is the binding constraint and a single GPU or a laptop is the target: Gemma 3, the small Qwen3 models, Mistral Small, Phi-4.

**Memory follows the total; check what fits.** DeepSeek-V3 in 8-bit weights is 671 GB, more than eight 80 GB GPUs hold, so it is served on a node of larger-memory GPUs or across two nodes. Kimi K2 needs about 1 TB at 8 bits. gpt-oss-120b ships its expert weights in a roughly 4-bit format, about 62 GB in total, which is why it fits on a single 80 GB GPU. A dense 27B model in bf16 needs 54 GB. The first question for any self-hosting plan is "what does the total parameter count cost in memory at the precision I will serve?".

**Speed follows the active count, until the batch is large.** At small batch, a 235B-A22B model decodes about as fast as a dense 22B model, because each token reads only the active experts. At large batch nearly every expert is used every step, and the system becomes limited by all-to-all communication between GPUs (Chapter 11, 11.2). MoE is cheapest per token when you have high, steady traffic to spread over many GPUs. It is a poor fit for a single-GPU, low-traffic deployment, where a dense model of the same *active* size would need a fraction of the memory.

**Tokens per parameter have gone far past Chinchilla.** Llama 3 8B saw about 15T tokens (about 1,900 per parameter). Qwen3's report gives about 36T tokens across sizes, and DeepSeek-V3 saw 14.8T tokens, which is 400 per *active* parameter. Small models are deliberately over-trained for inference economy (12.5), and the open question is data supply, not compute (Chapter 13, 13.6).

**Licences decide what you can do downstream.** Apache 2.0 and MIT allow commercial use, fine-tuning and, in general, distillation. The custom licences (Llama, Gemma, Kimi's modified MIT) add conditions: usage thresholds, attribution, acceptable-use policies, naming rules for derivatives. Some API terms forbid using a provider's outputs to train competing models. "Open weights" is not "open source": very few releases (OLMo is the standard example) publish training data and code, which is what makes a model reproducible and safe to use for contamination-sensitive research.

### Reasoning variants

Three packaging patterns exist, and an interviewer may ask you to choose between them:

- **A separate reasoning model.** DeepSeek-R1 next to V3, Magistral next to Mistral Small, Phi-4-reasoning next to Phi-4. This is the clearest to evaluate and serve, at the cost of two models to host.
- **A hybrid model with a thinking switch.** Qwen3's original release, later DeepSeek V3.x releases and GLM-4.5 use one set of weights, with a chat-template flag or tag that turns thinking on or off. This means one deployment, but the two modes can interfere during training, and Qwen later released separate Instruct and Thinking checkpoints for that reason.
- **A reasoning-only model with an effort setting.** gpt-oss has low, medium and high effort, which trade thinking tokens for accuracy at request time.

In all three, the reasoning tokens are billed and consume KV cache. A request that thinks for 5,000 tokens before a 200-token answer costs as much decode time as 26 such answers, so effort control and thinking budgets are cost controls as much as quality controls (Chapter 19, 19.8).

**Distilled small models.** DeepSeek released R1-distilled versions of Qwen2.5 and Llama 3 models from 1.5B to 70B parameters, trained by supervised fine-tuning on R1's reasoning traces, with no RL. The finding that matters for interviews: for small models, distilling traces from a strong reasoner beat running RL on the small model directly at comparable cost (Chapter 19, 19.9). Most small "thinking" models since then come from a mix of distillation and a short RL stage.

## Pitfalls

- Using raw BERT `[CLS]` or raw LLM hidden states as sentence embeddings without contrastive training, or mean-pooling a causal model's states.
- Comparing models on public leaderboards instead of your own labelled data and languages.
- Assuming MoE saves memory.
- Quoting Kaplan's conclusion (grow the model) as current practice, or quoting 14.6B as the Chinchilla optimum for 1e23 FLOPs (12.5).
- Picking a model by parameter count alone, ignoring context length, licence, latency and quality in the languages you serve.
- Costing a reasoning model by its answer length while ignoring the thinking tokens.
- Treating an advertised context length as a tested one.

## Interview questions

**Mid level**

**1. Compare BERT, GPT and T5: architecture, objective and best use.** BERT is an encoder with bidirectional attention trained by masked language modelling: every position sees both sides, which makes it strong for classification, extraction, reranking and (after contrastive fine-tuning) embeddings, but it cannot generate left to right. GPT is a decoder with a causal mask trained on next-token prediction: every position is a training example and generation is native, which is why all general LLMs are decoders. T5 is an encoder-decoder trained by span corruption, framing every task as text-to-text; it suits input-to-output tasks such as translation and summarisation. One number: BERT-base at 110M parameters costs about 14 GFLOPs per 64-token request against about 1,000 for an 8B decoder, a 73-fold difference that decides high-volume classification. The follow-up is "then why not always use BERT for classification?": because a decoder handles new labels, few-shot settings and long instructions without retraining, so it wins when labels change often or volume is low.

**2. Why is a vanilla BERT `[CLS]` vector a poor sentence embedding, and what fixes it?** Pretraining never asked `[CLS]` to summarise meaning so that cosine similarity is meaningful: next-sentence prediction was a weak signal, and RoBERTa dropped it. Raw BERT vectors are also anisotropic, crowded into a narrow cone, so all cosines are high and uninformative. The fix is contrastive fine-tuning on pairs (Sentence-BERT, then E5, GTE, BGE), usually with masked mean pooling, which pulls paraphrases together and pushes in-batch negatives apart. The follow-up is "what about decoder LLMs?": use last-token pooling or make attention bidirectional, add task instructions to queries, then train contrastively (12.2). Do not mean-pool a causal model's raw states, because early positions have seen only a prefix of the text.

**3. What is a mixture of experts and why is it attractive?** An MoE layer replaces the feed-forward block with many expert FFNs and a router that sends each token to the top k of them, so total parameters grow with the number of experts while compute per token grows only with k. At a fixed training budget it reaches lower loss than a dense model with the same active parameters, because it stores more knowledge per FLOP. One number: DeepSeek-V3 has 671B total and 37B active parameters, so each token costs the FLOPs of a 37B dense model (about 74 GFLOPs) but serving needs memory for all 671B. When not to use one: a single-GPU or low-traffic deployment, where memory binds and experts sit idle. The expected follow-up is "how do you stop the router collapsing?": an auxiliary load-balancing loss or a per-expert selection bias (Chapter 11, 11.2).

**4. How does CLIP learn a shared image-text space?** It trains an image encoder and a text encoder together on image-caption pairs with a symmetric InfoNCE loss. In a batch of N pairs, the N×N matrix of cosine similarities divided by a temperature goes through a cross-entropy whose correct class is the diagonal, applied over rows (image to text) and columns (text to image). Every other caption in the batch is a negative, so large batches help: CLIP used 32,768. The temperature is learned, initialised at 0.07 and with the logit scale clipped at 100. The follow-up is "how does SigLIP differ?": it replaces the batch softmax with an independent sigmoid per pair, which removes the batch-wide normalisation and works well at smaller batches (Chapter 14, 14.2).

**Senior level**

**5. State the Chinchilla result. Why do labs now train small models far beyond it?** For a fixed training budget C ≈ 6ND, loss is minimised by growing parameters and tokens in equal proportion, about 20 tokens per parameter; Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at the same compute. But Chinchilla minimises training compute only, and a deployed model pays about 2N FLOPs for every token it serves. When lifetime inference tokens exceed training tokens, a smaller model trained longer reaches the same loss at a lower total cost, which is why Llama 3 8B saw about 15T tokens (about 1,900 per parameter). Over-training has limits: returns diminish, high-quality data runs out (Chapter 13, 13.6), and heavily over-trained models degrade more under low-bit quantisation. The follow-up is "is 20 universal?": no. It is specific to that data, tokenizer and recipe, and must be refitted for your own setup.

**6. You have a budget of 1e23 FLOPs. How big a model and how many tokens? What if it will serve 1B requests a month?** With C = 6ND and D = 20N, C = 120N², so N = √(10²³/120) ≈ 29B parameters on about 580B tokens. The Besiroglu re-fit of Chinchilla's loss gives the same, 29.4B on 566B tokens; Chapter 13's 14.6B comes from the flawed approach-3 fit (12.5, and the "Three answers for one budget" passage in 13.3). For serving, assume 1B requests of 1,000 tokens each, so 10¹² tokens a month, and a year of service. The 29B model then spends 2 × 29.4B × 1.2 × 10¹³ ≈ 7.1 × 10²³ FLOPs on inference, seven times its training compute. Under the re-fitted loss, a 10.9B model needs 2.7T tokens (1.8 × 10²³ FLOPs) to reach the same loss but only 2.6 × 10²³ FLOPs of inference. That totals 4.4 × 10²³ against 8.1 × 10²³, 46% less (`code/ch12_check.py`). So train roughly a 10B model on several trillion tokens. Since inference FLOPs run at lower utilisation than training FLOPs, the real optimum is smaller still. The follow-up is "what stops you going to 3B?": the token count needed explodes, because the data term falls slowly, and data supply becomes the limit.

**7. Design the model choice for tagging 50M catalogue items with 300 attributes, in English and a second, lower-resource language.** Start with arithmetic. At 200 tokens per item, 50M items is 10¹⁰ tokens; a large API model at $1 per million input tokens costs about $10,000 per pass before output tokens, and every relabel repeats it. A fine-tuned encoder runs the same pass on a few GPUs for under a day. So: label 5,000–10,000 items with a large LLM, with human review of a stratified sample per language. Fine-tune a multilingual encoder (XLM-R or mDeBERTa) with a 300-way multi-label sigmoid head, or a small LLM if attributes need reasoning over long text. Use a VLM only for items whose information is mainly in images. Route low-confidence items to the large model and then to humans, and report per-language, per-attribute F1 with confidence intervals, since rare attributes and the low-resource language are where it fails. The trade-off: the encoder must be retrained when attributes change, while the LLM path adapts by prompt. The follow-up is "how do you set the confidence threshold?": on held-out data, by plotting precision of the kept items against the escalation rate (12.6).

**8. Are emergent abilities real?** Wei et al. (2022) reported abilities that are near chance below some scale and then jump. Schaeffer et al. (2023) showed that much of this comes from the metric. If per-token accuracy p rises smoothly with scale, exact match on a 10-token answer is p¹⁰, which stays near zero and then climbs steeply. In the toy curve in `code/ch12_check.py`, per-token accuracy goes 0.50 → 0.65 → 0.77 between 10^9.5 and 10^10.5 parameters while exact match goes 0.001 → 0.013 → 0.072, which looks like a sudden jump. With log-likelihood or token-level metrics, progress is smooth and predictable. Du et al. (2024) reconcile the views: across model sizes and token counts, many abilities appear once *pretraining loss* falls below a threshold, so they are sudden as a function of loss but predictable, because loss is. My answer: the underlying competence improves smoothly, but some user-visible abilities really are thresholded, because a task fails if any step fails. The follow-up is "so can you forecast them?": partly, by predicting loss from a scaling law, then mapping loss to accuracy with a fitted sigmoid (Chapter 13, 13.8).

**9. A model card says "235B-A22B, 128k context, Apache 2.0". What does it cost to serve, and when would you prefer a dense 32B?** 235B total parameters need 470 GB in bf16 or 235 GB at 8 bits, so at least four 80 GB GPUs at 8 bits before any KV cache. Each token does the work of 22B active parameters, about 44 GFLOPs, so per-token compute is close to a dense 22B model. At small batch it also decodes almost as fast as one, because only the active experts' weights are read. Apache 2.0 allows commercial use and fine-tuning. Prefer a dense 32B (64 GB in bf16, one or two GPUs) when traffic is low or bursty, when you will fine-tune heavily (MoE fine-tuning is less stable and needs expert-parallel tooling), or when you serve on one GPU. The follow-up is "why does the MoE's advantage shrink at high batch?": with 128 experts and top-8 routing, 64 tokens in a batch touch about 126 experts, so a step reads nearly all weights and all-to-all traffic dominates (Chapter 11, 11.2).

**10. Your team wants reasoning for a customer-facing assistant. Separate reasoning model, hybrid thinking mode, or an effort setting?** The choice is about cost control and operational simplicity, since quality differences between the packagings are smaller than differences between model families. A hybrid or effort-controlled model needs one deployment and lets you route easy requests to non-thinking mode and hard ones to thinking. A separate reasoning model is easier to evaluate and pin, but doubles hosting. The number to compute: if 20% of requests think for 4,000 tokens before a 300-token answer, average decode length rises from 300 to 1,100 tokens, about 3.7 times the decode cost and TPOT-bound latency. When not to use reasoning: latency-critical paths with simple intents, where a router plus a non-thinking model wins. The follow-up is "how do you decide which requests get thinking?": train or prompt a cheap classifier on logged requests labelled by whether thinking changed the outcome, and evaluate the routed system end to end.

## Practice

1. Fine-tune XLM-R or a multilingual DeBERTa on a public multilingual sentiment or topic dataset in two languages, then compare with a few-shot LLM on cost per 1,000 items and accuracy, per language.

<details><summary>Hint and expected result</summary>

Use the same test split for both. Expect the fine-tuned encoder to match or beat the few-shot LLM on the higher-resource language once it has a few thousand labels, at well over 10× lower cost per 1,000 items; the gap usually narrows or reverses for the lower-resource language and for rare classes. Report accuracy with a confidence interval (Chapter 23): on a 500-item test set, ±4 points at 70% accuracy is normal noise.

</details>

2. Run CLIP zero-shot on 50 photos from a public image dataset with labels like "a photo of a {label}". Where does it fail?

<details><summary>Hint and expected result</summary>

Expect high accuracy on common, visually distinct classes and failures on fine-grained classes (similar species or products), on text-heavy images, on counting and on unusual viewpoints. Try prompt ensembling ("a photo of a {label}", "a close-up of a {label}"); it usually adds a few points.

</details>

3. Redo the Chinchilla arithmetic for 1e21, 1e23 and 1e24 FLOPs without a calculator, then explain in two sentences why Chapter 13 gives 14.6B for 1e23.

<details><summary>Hint and expected result</summary>

N = √(C/120): about 2.9B, 29B and 91B parameters, with 20 times as many tokens. √(10²³/120) = √(8.3 × 10²⁰) ≈ 2.9 × 10¹⁰. The 14.6B comes from the approach-3 constants, which imply 78 tokens per parameter and which Besiroglu et al. showed to be a flawed fit (12.5).

</details>

4. Take the MoE layer from Chapter 11 (11.2), add a capacity factor that drops overflow tokens, and measure how many are dropped with and without the auxiliary loss.

<details><summary>Hint and expected result</summary>

Capacity per expert = capacity factor × tokens × k / E, and the dropped share is the sum of max(0, load − CF/E) over experts. The skewed router in `code/ch11_check.py` drops 37% of routed tokens at a capacity factor of 1.0 and 28% at 1.25; after balancing (loads 0.123–0.127) it drops under 1% at 1.0. Expect your trained router to land between those extremes without balancing and near the balanced figure with it.

</details>

5. Pick two open models from 12.9 and fill in the 12.8 checklist from their cards. Compute the serving memory at bf16 and 8 bits, and the per-token decode FLOPs.

<details><summary>Hint and expected result</summary>

Memory = total parameters × bytes per parameter; FLOPs per token ≈ 2 × active parameters. For Qwen3-30B-A3B (about 30.5B total, 3.3B active): about 61 GB in bf16, 31 GB at 8 bits, about 7 GFLOPs per token, so it runs on a single 80 GB GPU with room for cache, while decoding with the compute of a 3B model.

</details>

6. **Research exercise.** Reproduce the "emergence is a metric artefact" result. Take one model family at 4–6 sizes (for example Qwen3 0.6B to 32B), and a task with multi-token exact answers (3- to 5-digit addition or multiplication, 200 items). Measure exact-match accuracy, per-token accuracy and the log-likelihood of the correct answer. Plot all three against log parameters, with bootstrap confidence intervals over items.

<details><summary>Hint and expected result</summary>

Exact match should look like a sharp jump at one size, while per-token accuracy and log-likelihood rise smoothly; exact match should be roughly per-token accuracy raised to the answer length. The toy version in `code/ch12_check.py` shows the shape. If your exact-match curve is smooth too, the task is probably too easy for the smallest model; lengthen the answers. Bonus: plot accuracy against each model's validation loss instead of size (the Du et al. view).

</details>

## Read next

- Devlin et al., "BERT" (2019); Liu et al., "RoBERTa" (2019).
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2020).
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019); Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020).
- Reimers and Gurevych, "Sentence-BERT" (2019); BehnamGhader et al., "LLM2Vec" (2024).
- Fedus, Zoph and Shazeer, "Switch Transformers" (2021); Jiang et al., "Mixtral of Experts" (2024).
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021); Liu et al., "Visual Instruction Tuning" (LLaVA, 2023).
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020); Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022); Besiroglu et al., "Chinchilla Scaling: A replication attempt" (2024).
- Schaeffer, Miranda and Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?" (2023); Du et al., "Understanding Emergent Abilities of Language Models from the Loss Perspective" (2024).
- The technical reports of DeepSeek-V3, DeepSeek-R1, Qwen3 and Gemma 3, read with the 12.8 checklist in hand.
- Links to most papers above are in the Key concepts section at the top of this chapter.

## Done when

- Without notes, draw the three attention masks and name the family, objective and best use of each in two minutes.
- Explain in one minute why a causal model needs last-token pooling or a bidirectional conversion to be a good embedder.
- Given "671B total, 37B active", state the bf16 and 8-bit serving memory (1.34 TB, 671 GB) and the forward FLOPs per token (about 74 GFLOPs) in under a minute.
- Compute the Chinchilla optimum for 1e23 FLOPs by hand (≈29B parameters, ≈580B tokens) and explain the 14.6B figure in Chapter 13.
- Estimate decode throughput for an 8B bf16 model at batch 64 from memory bandwidth alone (≈13k tokens/s at 3.35 TB/s, before the KV cache).
- Compute the cost of a cascade with c_s = 0.05·c_L at 20% escalation (0.25 of large-only), and its break-even escalation rate (95%).
- Place any new open model on the 12.9 map from its card in five minutes: dense or MoE, total and active parameters, context, licence, reasoning packaging.
- Justify a model choice for a classification or extraction workload at 10M items a month in terms of quality, latency, cost and language.

---

[← 11 · Architectures beyond the vanilla transformer](11-architectures-beyond-the-transformer.md) · [Contents](../README.md#contents) · [13 · Scaling laws →](13-scaling-laws.md)
