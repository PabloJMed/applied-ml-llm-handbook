# 14 · Multimodal models

Part III · Transformers and architectures. Assumes Chapter 10 (transformers, RoPE) and Chapter 12 (CLIP and contrastive learning). Chapter 29 covers embeddings for retrieval, including image–text embeddings; Chapter 23 covers how to evaluate models in general.

## Why it matters

Frontier models read images, documents, screenshots, audio and video, and increasingly produce images and speech. For a research scientist, multimodality is mostly a question of **interfaces**: how a non-text signal becomes a sequence of vectors a transformer can use, how many of those vectors it costs, what data teaches the model to ground language in pixels or sound, and how to tell real visual understanding from a strong language prior. Those are the questions interviews ask, and the ones this chapter answers.

Numbers in this chapter come from `code/ch14_check.py` (CPU, seconds).

## Key concepts

- **Vision Transformer (ViT)**: split an image into patches, embed each patch as a token. [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929).
- **Contrastive image–text pretraining**: CLIP (softmax) and SigLIP (sigmoid) losses. [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343).
- **Vision–language models (VLMs)**: a vision encoder, a projector and an LLM. [LLaVA](https://arxiv.org/abs/2304.08485), [Flamingo](https://arxiv.org/abs/2204.14198).
- **Native-resolution vision and 2-D positions**: variable image sizes with two-dimensional rotary positions. [Qwen2-VL](https://arxiv.org/abs/2409.12191).
- **Early fusion**: images as discrete tokens in one shared vocabulary. [Chameleon](https://arxiv.org/abs/2405.09818), [Transfusion](https://arxiv.org/abs/2408.11039).
- **Speech**: encoder–decoder recognition and audio tokens. [Whisper](https://arxiv.org/abs/2212.04356).
- **Diffusion and flow matching** for image generation. [DDPM](https://arxiv.org/abs/2006.11239), [Latent diffusion](https://arxiv.org/abs/2112.10752), [Classifier-free guidance](https://arxiv.org/abs/2207.12598).

## 14.1 Four ways to combine modalities

| Design | How images enter | Examples | Trade-off |
|---|---|---|---|
| Encoder + projector ("late fusion") | A pretrained vision encoder's patch features, mapped into the LLM's embedding space and placed in the token sequence | LLaVA family, most open VLMs | Simple and data-efficient; understanding only, unless a separate generator is added |
| Cross-attention | The LLM attends to vision features through extra cross-attention layers | Flamingo, Llama 3.2 Vision | Text path unchanged; more parameters; harder to interleave many images |
| Early fusion, discrete | Images tokenised into codes from a learned codebook, in the same vocabulary as text | Chameleon | One model reads and writes images; tokenisation loses detail |
| Unified, continuous | Text by next-token prediction, images by diffusion, in one transformer | Transfusion and successors | Strong generation and understanding; complex training |

Most open understanding models use the first design, and the rest of this chapter starts there.

## 14.2 Vision encoders

**Patches as tokens.** A ViT cuts an image into p × p patches, flattens each and maps it linearly to the model width. A 224 × 224 image with 16-pixel patches becomes 196 tokens of 768 raw values each. Positions are added (learned or 2-D sinusoidal), then a standard transformer encoder runs over the patches.

**Contrastive pretraining.** CLIP trains an image encoder and a text encoder so that matching image–caption pairs have high cosine similarity and mismatched pairs low, with a softmax over the batch in both directions (Chapter 12). **SigLIP** replaces the softmax with an independent sigmoid per pair:

```python
logits = t * img @ txt.T + b                    # learnable temperature t and bias b
labels = 2 * torch.eye(B) - 1                   # +1 for matched pairs, -1 for all others
loss = -F.logsigmoid(labels * logits).sum() / B
```

Each pair is a separate binary decision, so no batch-wide normalisation is needed, which makes very large batches cheaper and smaller batches work better. SigLIP-family encoders are now the most common vision towers in open VLMs. (The two losses are on different scales, 0.039 and 5.02 on the toy batch in the check script; compare them only within one method.)

Why contrastive encoders for VLMs: their features are already aligned with language, which makes the projector's job easy. Their weakness: trained to match whole captions, they under-represent fine detail (small text, counting, spatial relations), which VLM training then has to recover.

## 14.3 Connecting vision to an LLM

**The LLaVA recipe.** Run the vision encoder, take its patch features (576 tokens from a 336-pixel image with 14-pixel patches), map them with a small MLP projector to the LLM's width, and splice them into the text sequence where the image appears. The LLM then trains with the usual next-token loss on the answer tokens only:

```python
img_emb = projector(vision_encoder(image))      # (1, 576, d_llm); projector: 6.3M params here
seq = torch.cat([text_before, img_emb, text_after], dim=1)
labels[...] = -100                              # no loss on the prompt or the image tokens
labels[answer_positions] = answer_ids
```

**The token budget is the central constraint.** Detail needs resolution, and tokens grow with the square of the side:

| Image side (14-px patches) | Vision tokens | After 2 × 2 merging |
|---|---|---|
| 224 | 256 | 64 |
| 448 | 1,024 | 256 |
| 896 | 4,096 | 1,024 |
| 1,344 | 9,216 | 2,304 |

A document page needs high resolution to read small print; a photo of a cat does not. Three techniques manage this:

- **Tiling**: cut a large image into encoder-sized tiles plus a downscaled overview (LLaVA-NeXT and others).
- **Native resolution**: encode the image at its own aspect ratio and size, with **2-D rotary positions** (half the rotary dimensions encode the row, half the column), so the encoder handles any shape (Qwen2-VL and later). The check script confirms that two patch pairs with the same (row, column) offset get the same attention score.
- **Token compression**: merge each 2 × 2 block of neighbouring tokens into one with 4× the channels (pixel shuffle), cutting 576 tokens to 144, or use a small cross-attention "resampler" with a fixed number of learned queries.

## 14.4 Training a VLM

A typical open recipe has three or four stages:

1. **Alignment**: freeze the vision encoder and LLM, train only the projector on image–caption pairs. Cheap; teaches the LLM to read the projected features.
2. **Pretraining on mixed data** (increasingly common): unfreeze more, train on large interleaved image–text documents, OCR data, charts, diagrams and captions, mixed with text-only data so language ability does not degrade.
3. **Instruction tuning**: visual question answering, document and chart questions, grounding (point to or box an object), multi-image and video conversations, with the LLM unfrozen.
4. **Preference and RL training**: reduce hallucinated objects and improve reasoning on visual maths and charts, with verifiable rewards where possible (Chapter 19).

**Data matters more than architecture.** Most gains in open VLMs have come from better data: dense, accurate re-captioning of images by strong models, synthetic charts and documents with known answers, large OCR corpora, and careful mixing. Keep text-only data in every stage, or the LLM's language and reasoning skills degrade.

## 14.5 Audio and speech

**Recognition.** Whisper converts 30 seconds of 16 kHz audio into a log-mel spectrogram with a 10 ms hop (3,000 frames), downsamples by 2 with a convolution (1,500 encoder positions) and decodes text with a transformer decoder trained on 680k hours of weakly labelled audio. Audio LLMs reuse the same pattern as VLMs: an audio encoder, a projector, an LLM.

**Audio as tokens.** Neural audio codecs compress audio into discrete codes from several codebooks per frame (residual vector quantisation). At 12.5 frames per second with 8 codebooks, 30 seconds is 3,000 tokens. An LLM that reads and writes these tokens can listen and speak directly ("speech-to-speech"), which preserves tone and allows low-latency spoken conversation. The design problems are interleaving text and audio streams, latency (the model must start speaking before it has "thought"), and keeping reasoning quality close to the text model's.

**Vector quantisation** (images and audio): map each continuous vector to its nearest codebook entry; pass gradients through the non-differentiable lookup with the straight-through trick and add a commitment loss so the encoder output stays near its code:

```python
idx = torch.cdist(z, codebook).argmin(-1)
zq = codebook[idx]
z_st = z + (zq - z).detach()          # forward: zq; backward: gradient flows to z
commit = F.mse_loss(z, zq.detach())
```

## 14.6 Generating images

**Diffusion.** Train a network to undo noise. The forward process mixes an image x₀ with Gaussian noise: x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε. With the standard linear schedule over 1,000 steps, the signal weight is 1.000 at t = 0, 0.722 at t = 250, 0.279 at t = 500 and 0.006 at t = 999. The network predicts ε from (x_t, t); sampling starts from pure noise and removes predicted noise step by step. **Latent diffusion** runs the process in the compressed latent space of an autoencoder, which is far cheaper. **Flow matching** learns a velocity field along straight paths from noise to data and samples in fewer steps; most recent image and video generators use it with transformer backbones (diffusion transformers).

**Classifier-free guidance.** Train the same network with and without the text condition (by randomly dropping it), then at sampling time extrapolate: ε̂ = ε_uncond + w (ε_cond − ε_uncond), with w around 3–7. Higher w follows the prompt more closely at the cost of diversity.

**Autoregressive image generation.** Generate image tokens from a VQ tokenizer one by one, like text (Chameleon), or predict tokens at progressively finer scales. Unified models that use next-token prediction for text and diffusion for images in one transformer (Transfusion) are an active line of work, because one model that both understands and generates can use the same knowledge for both.

## 14.7 Video

Video is images over time, and the token count explodes: 10 minutes at 2 frames per second with 256 tokens per frame is 307,200 tokens. Standard tools: sample frames sparsely (and adaptively, more where the scene changes), merge tokens across time as well as space, use 3-D (time, row, column) rotary positions, and process long videos with memory or retrieval over clips. Evaluate on questions that need the whole video, not a single frame, or the model will learn to answer from one lucky frame.

## 14.8 Evaluating multimodal models

| What to test | Examples | Watch for |
|---|---|---|
| General visual reasoning | MMMU, MMBench | Many questions answerable from text alone |
| Documents, charts, OCR | DocVQA, ChartQA, OCRBench | Resolution and tiling settings change scores a lot |
| Maths in images | MathVista, MathVerse | Leakage of popular problems |
| Hallucination | POPE, object-presence probes | "Yes" bias; rate the model on images where the object is absent |
| Video | Video-MME, long-video QA | Answerable from a single frame |
| Speech | Word error rate by accent and noise | Text normalisation differences between systems |

Two checks a research scientist should always run:

- **Blind baseline**: run the benchmark with the image removed or replaced by noise. If accuracy stays well above chance, the benchmark (or the model) is relying on language priors, and the gap to the blind score is the real visual contribution.
- **Resolution sweep**: evaluate at several image resolutions and token budgets. A model "better at charts" may simply be using four times the tokens.

## Pitfalls

- Comparing VLMs evaluated at different resolutions or token budgets.
- Training only on caption data: the model learns to describe, not to answer questions about details.
- Dropping text-only data during multimodal training and losing language ability.
- Trusting a benchmark without a blind baseline.
- Forgetting that every image costs hundreds to thousands of tokens of context and prefill compute.
- Using a contrastive encoder at low resolution and expecting it to read small text.

## Interview questions

**1. How does an image become tokens in a ViT?** Cut it into p × p patches, flatten and linearly project each, add positions. 224² with 16-pixel patches gives 196 tokens.

**2. Compare CLIP and SigLIP losses.** CLIP: softmax cross-entropy over the batch in both directions. SigLIP: an independent sigmoid per image–text pair with a learned bias. SigLIP needs no batch-wide normalisation, so it scales to large batches cheaply and works better at small ones.

**3. Describe the LLaVA architecture and training.** Frozen vision encoder, MLP projector into the LLM's embedding space, image tokens spliced into the sequence. Stage 1 trains the projector on captions; stage 2 instruction-tunes with the LLM unfrozen.

**4. How do VLMs handle high-resolution documents?** Tiling with an overview image, or native-resolution encoding with 2-D RoPE; then token merging (pixel shuffle) or a resampler to fit the token budget.

**5. Cross-attention versus projector-in-sequence designs?** Cross-attention leaves the LLM's text path untouched and keeps image features out of the context; projector designs are simpler, reuse the LLM fully and handle interleaving naturally, at the cost of context length.

**6. What is early fusion, and why is it attractive?** Images become discrete tokens in the same vocabulary, so one model reads and generates both. Costs: the tokenizer loses detail and training is harder to stabilise.

**7. How would you detect that a VLM relies on language priors?** Blind baselines (image removed), counterfactual images that contradict the prior, hallucination probes for absent objects.

**8. Explain classifier-free guidance.** Train with the condition randomly dropped; at sampling, push the prediction away from the unconditional one: ε_u + w(ε_c − ε_u).

**9. How does Whisper process audio?** Log-mel spectrogram (10 ms hop), conv downsampling, transformer encoder; a decoder generates text with special tokens for language, task and timestamps.

**10. What limits video understanding?** Token count. Sparse and adaptive frame sampling, spatio-temporal token merging, 3-D positions, and memory or retrieval for long videos.

## Practice

1. Build a tiny VLM: a frozen small ViT or SigLIP encoder, an MLP projector, a small LLM. Train the projector on a caption dataset, then instruction-tune on a VQA subset.
2. Implement pixel-shuffle merging and measure accuracy against tokens on a document-QA subset.
3. **Research exercise.** Take two open VLMs and a benchmark. Score each with the real image, with no image, and with a mismatched image. Report the visual contribution (real minus blind) with confidence intervals (Chapter 23), and discuss what the benchmark measures.
4. Implement the DDPM forward process and a small denoiser on MNIST; add classifier-free guidance and vary w.
5. Compute the token cost of a 20-page PDF at three resolutions for a model you use.

## Read next

- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020); Radford et al., CLIP (2021); Zhai et al., SigLIP (2023).
- Liu et al., "Visual Instruction Tuning" (LLaVA, 2023), and the LLaVA-NeXT notes.
- Alayrac et al., "Flamingo" (2022).
- Wang et al., "Qwen2-VL" (2024), for native resolution and M-RoPE.
- Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models" (2024); Zhou et al., "Transfusion" (2024).
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022).
- Ho, Jain and Abbeel, "Denoising Diffusion Probabilistic Models" (2020); Rombach et al., "Latent Diffusion" (2022); Ho and Salimans, "Classifier-Free Diffusion Guidance" (2022).
- Links to these papers are in the Key concepts section at the top of this chapter.

## Done when

You can explain how images, audio and video become tokens and compute their token counts. You can compare CLIP and SigLIP, and the four ways of combining modalities. You can describe a VLM training recipe and why data quality dominates.

You can explain tiling, native resolution with 2-D RoPE and token merging. You can explain diffusion, flow matching and classifier-free guidance at whiteboard level. You can design a multimodal evaluation with blind baselines and resolution controls.

---

[← 13 · Scaling laws](13-scaling-laws.md) · [Contents](../README.md#contents) · [15 · How LLMs are trained →](15-how-llms-are-trained.md)
