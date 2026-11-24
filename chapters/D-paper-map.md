# D · Paper map

About 126 papers that come up again and again in LLM research, applied ML and interviews, grouped by theme, one paragraph each: the problem it solved, the key idea, and why it still matters. Entries 1–40 cover foundations and applied work; 41–119 cover the research areas of the LLM-scientist chapters; 120–125 are classics of RLHF, inference and code evaluation, placed in their themes. The list covers papers to the end of 2025 and was last checked in November 2026. For the core papers, a *Know cold* note lists the facts interviewers probe: the numbers, the design choices and the reasons behind them. Knowing what a paper did is the acceptable answer; knowing why its choices were made and what later work corrected is the strong one. Every title links to the paper. Read the abstract, the figures and the conclusion first; go deeper where your role needs it. A star (★) marks ten to know cold for any applied LLM role; the research core is listed below.

**The research-scientist core (know these cold):** 10 (transformer), 16, 17 and 56 (Kaplan, Chinchilla and its replication), 22 and 24 (InstructGPT, DPO), 26 and 76 (R1, GRPO), 120 (RL from human preferences), 41 (µP), 47–48 (DeepSeekMoE, MLA), 61 (FineWeb), 66–67 (ZeRO, Megatron), 71 (reward over-optimisation), 89 (error bars for evals), 95 and 98 (circuits, sparse autoencoders), 101 (weak-to-strong), 107–108 (Llama 3, DeepSeek-V3).

## Foundations of deep learning

**1. Rumelhart, Hinton and Williams,** [**"Learning representations by back-propagating errors"**](https://www.nature.com/articles/323533a0) **(1986).** Showed that the chain rule, applied layer by layer, trains multi-layer networks that learn useful internal features. Every model in this guide is trained this way (Ch 5).

**2. Kingma and Ba,** [**"Adam: A Method for Stochastic Optimization"**](https://arxiv.org/abs/1412.6980) **(2015); Loshchilov and Hutter,** [**"Decoupled Weight Decay Regularization"**](https://arxiv.org/abs/1711.05101) **(AdamW, 2019).** Momentum plus per-parameter step sizes with bias correction; AdamW separates weight decay and is the default for transformers (Ch 5).

**3. Srivastava et al.,** [**"Dropout"**](https://jmlr.org/papers/v15/srivastava14a.html) **(2014).** Randomly zeroing units during training stops co-adaptation and acts like averaging many thinned networks (Ch 5).

**4. Ioffe and Szegedy,** [**"Batch Normalization"**](https://arxiv.org/abs/1502.03167) **(2015); Ba, Kiros and Hinton,** [**"Layer Normalization"**](https://arxiv.org/abs/1607.06450) **(2016).** Normalising activations made deep networks train faster and more stably; transformers use layer norm because it does not depend on the batch (Ch 5, 10).

**5. He et al.,** [**"Deep Residual Learning for Image Recognition"**](https://arxiv.org/abs/1512.03385) **(2016).** Skip connections let gradients flow through very deep networks. The residual stream is the backbone of the transformer (Ch 5, 10).

## Representations and sequence models

**6. Mikolov et al.,** [**"Distributed Representations of Words and Phrases and their Compositionality"**](https://arxiv.org/abs/1310.4546) **(2013).** Skip-gram with negative sampling made word embeddings cheap; the same trick embeds items from baskets and sessions (Ch 8).

**7. Hochreiter and Schmidhuber,** [**"Long Short-Term Memory"**](https://doi.org/10.1162/neco.1997.9.8.1735) **(1997).** A gated cell updated additively lets gradients survive long sequences (Ch 8).

**8. Bahdanau, Cho and Bengio,** [**"Neural Machine Translation by Jointly Learning to Align and Translate"**](https://arxiv.org/abs/1409.0473) **(2014).** Attention removed the single-vector bottleneck of seq2seq: the direct ancestor of the transformer (Ch 8).

**9. Sennrich, Haddow and Birch,** [**"Neural Machine Translation of Rare Words with Subword Units"**](https://arxiv.org/abs/1508.07909) **(2016).** Brought byte-pair encoding to NLP; every modern tokenizer descends from it (Ch 9).

## Transformers and pre-trained models

**10. ★ Vaswani et al.,** [**"Attention Is All You Need"**](https://arxiv.org/abs/1706.03762) **(2017).** Replaced recurrence with multi-head self-attention, positional encodings, residuals and feed-forward blocks, making training parallel. Be able to implement the block and justify each design choice (Ch 10). *Know cold:* the base model has d_model = 512, 8 heads, d_ff = 4·d_model and six encoder plus six decoder layers; it used post-norm, which modern LLMs replaced with pre-norm for stability at depth; scores are divided by √d_k, and training used an inverse-square-root learning-rate schedule with 4,000 warm-up steps and label smoothing of 0.1.

**11. ★ Devlin et al.,** [**"BERT"**](https://arxiv.org/abs/1810.04805) **(2019).** A bidirectional encoder pre-trained with masked language modelling, then fine-tuned; still the right family for high-volume classification and reranking (Ch 12, 41). *Know cold:* 15% of tokens are selected for prediction, and of those 80% become [MASK], 10% a random token and 10% stay unchanged, so the model cannot rely on seeing [MASK] at fine-tuning time; base has 110M parameters and large 340M; the next-sentence-prediction objective was later shown unnecessary (RoBERTa).

**12. Radford et al.,** [**"Language Models are Unsupervised Multitask Learners"**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) **(GPT-2, 2019) and Brown et al.,** [**"Language Models are Few-Shot Learners"**](https://arxiv.org/abs/2005.14165) **(GPT-3, 2020).** Scaling a decoder-only model produced zero- and few-shot ability from prompts alone (Ch 12, 22).

**13. Raffel et al.,** [**"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**](https://arxiv.org/abs/1910.10683) **(T5, 2020).** Every task as text-to-text, plus a model study of objectives and data (Ch 12).

**14. Reimers and Gurevych,** [**"Sentence-BERT"**](https://arxiv.org/abs/1908.10084) **(2019).** Siamese fine-tuning made encoder embeddings work for semantic search at scale (Ch 29, 12).

**15. Radford et al.,** [**"Learning Transferable Visual Models From Natural Language Supervision"**](https://arxiv.org/abs/2103.00020) **(CLIP, 2021).** Contrastive training on image–caption pairs created a shared image–text space with zero-shot classification (Ch 14).

## Scaling and efficient models

**16. Kaplan et al.,** [**"Scaling Laws for Neural Language Models"**](https://arxiv.org/abs/2001.08361) **(2020).** Loss follows smooth power laws in parameters, data and compute (Ch 13). *Know cold:* Kaplan fitted N_opt ∝ C^0.73, putting most extra compute into parameters; Chinchilla found roughly equal exponents (about 0.5 each). The gap came mostly from using one learning-rate schedule length for runs of different token counts (which under-rates small models trained longer) and from counting non-embedding parameters at small scale.

**17. ★ Hoffmann et al.,** [**"Training Compute-Optimal Large Language Models"**](https://arxiv.org/abs/2203.15556) **(Chinchilla, 2022).** For fixed compute, scale parameters and tokens equally, about 20 tokens per parameter; know C ≈ 6ND and why practice now overtrains small models (Ch 13). *Know cold:* three approaches agreed roughly: the lower envelope of training curves at fixed model sizes, IsoFLOP profiles (loss against size at fixed compute), and a parametric fit L = E + A/N^α + B/D^β. Chinchilla (70B parameters, 1.4T tokens) used the same compute as Gopher (280B, 300B tokens) and beat it. The 20 tokens per parameter minimises training compute only; the third approach's fit was later found to be flawed (entry 56).

**18. Fedus, Zoph and Shazeer,** [**"Switch Transformers"**](https://arxiv.org/abs/2101.03961) **(2021).** Top-1 mixture-of-experts routing with a load-balancing loss: more parameters, same compute per token (Ch 11).

**19. Dao et al.,** [**"FlashAttention"**](https://arxiv.org/abs/2205.14135) **(2022).** Exact attention computed in tiles to cut memory traffic; hardware-aware engineering as a source of speed (Ch 10).

**20. Ainslie et al.,** [**"GQA"**](https://arxiv.org/abs/2305.13245) **(2023).** Shared key-value heads shrink the KV cache with little quality loss (Ch 10, 21).

## Training LLMs to follow instructions

**21. Wei et al.,** [**"Finetuned Language Models Are Zero-Shot Learners"**](https://arxiv.org/abs/2109.01652) **(FLAN, 2022).** Instruction tuning on many tasks improved zero-shot performance on unseen tasks (Ch 12, 18).

**22. ★ Ouyang et al.,** [**"Training language models to follow instructions with human feedback"**](https://arxiv.org/abs/2203.02155) **(InstructGPT, 2022).** The SFT → reward model → PPO recipe with a KL penalty (Ch 18). *Know cold:* labellers preferred outputs of the 1.3B InstructGPT model to those of the 175B GPT-3; the three stages used roughly 13k prompts for SFT, 33k for the reward model and 31k for PPO, with a 6B reward model; mixing pretraining gradients into PPO (PPO-ptx) reduced the "alignment tax" on public NLP benchmarks.

**23. Bai et al.,** [**"Constitutional AI: Harmlessness from AI Feedback"**](https://arxiv.org/abs/2212.08073) **(2022).** Self-critique against written principles, and AI-labelled preferences; the basis of RLAIF and LLM-as-judge thinking (Ch 27, 18, 24).

**24. ★ Rafailov et al.,** [**"Direct Preference Optimization"**](https://arxiv.org/abs/2305.18290) **(2023).** KL-constrained RLHF rewritten as a classification loss on preference pairs; no reward model, no RL loop (Ch 18). *Know cold:* the implicit reward is β log(π/π_ref), defined up to a prompt-dependent constant that cancels in the Bradley–Terry difference; under that model DPO has the same optimum as KL-regularised RLHF, but it is trained offline on fixed pairs, so it cannot explore; the gradient weights each pair by σ of the wrongly ordered implicit-reward margin (Appendix C), and the original experiments were sentiment control, TL;DR summarisation and Anthropic HH dialogue.

**25. Hinton, Vinyals and Dean,** [**"Distilling the Knowledge in a Neural Network"**](https://arxiv.org/abs/1503.02531) **(2015).** A student trained on a teacher's softened outputs; the route from an expensive model to a cheap one (Ch 15, 19, 41).

**26. DeepSeek-AI,** [**"DeepSeek-R1"**](https://arxiv.org/abs/2501.12948) **(2025).** RL with verifiable rewards produced long reasoning; the RL algorithm is GRPO (entry 76) (Ch 19). *Know cold:* R1-Zero is pure RL from the base model with rule-based accuracy and format rewards and no SFT: response length and accuracy rose together (AIME 2024 pass@1 from 15.6% to 71%), but outputs mixed languages and read poorly. R1 adds a cold start (SFT on thousands of curated long chains of thought), then reasoning RL with a language-consistency reward, then rejection-sampled SFT on about 800k examples and a final RL stage for all tasks. Distilling R1's outputs into smaller dense models beat running RL on those models directly.

**120. Christiano et al.,** [**"Deep Reinforcement Learning from Human Preferences"**](https://arxiv.org/abs/1706.03741) **(2017).** Learned a reward model from human comparisons of short trajectory clips and trained an RL agent on it, solving Atari and simulated-robotics tasks with feedback on under 1% of the agent's interactions: the origin of the reward-model-plus-RL recipe (Ch 18). *Know cold:* the reward model is fitted with the Bradley–Terry likelihood on pairwise comparisons; labels are collected asynchronously while the policy trains, so the reward model keeps up with the policy's new behaviour; and an agent optimising a frozen reward model learned to exploit it, the first clear reward hacking of a learned reward.

**121. Stiennon et al.,** [**"Learning to Summarize from Human Feedback"**](https://arxiv.org/abs/2009.01325) **(2020).** The recipe of entry 120 applied to language: SFT, a reward model on human comparisons of summaries, then PPO with a KL penalty to the SFT model (Ch 18). *Know cold:* summaries from the RLHF model were preferred to the human-written reference summaries on TL;DR; optimising the reward model too hard made summaries worse by human judgement even as reward rose, which is why the KL penalty exists; and the paper put label quality first, with close work alongside labellers.

**122. Touvron et al.,** [**"Llama 2: Open Foundation and Fine-Tuned Chat Models"**](https://arxiv.org/abs/2307.09288) **(2023).** The most detailed public RLHF report of its time (Ch 18). *Know cold:* separate helpfulness and safety reward models; iterative rounds that combined rejection sampling (best-of-n fine-tuning) with PPO; a margin term in the reward-model loss for pairs labelled "significantly better"; and Ghost Attention, which trains a system instruction to persist across turns.

## Fine-tuning efficiently

**27. Li and Liang,** [**"Prefix-Tuning"**](https://arxiv.org/abs/2101.00190) **(2021); Lester, Al-Rfou and Constant,** [**"The Power of Scale for Parameter-Efficient Prompt Tuning"**](https://arxiv.org/abs/2104.08691) **(2021).** Train only continuous prompt vectors on a frozen model; prompt tuning approaches full fine-tuning as models grow (Ch 20).

**28. ★ Hu et al.,** [**"LoRA: Low-Rank Adaptation of Large Language Models"**](https://arxiv.org/abs/2106.09685) **(2021).** Low-rank updates added to frozen weights; tiny trainable parameter count and no added latency once merged (Ch 20). *Know cold:* B is initialised to zero (and A randomly), so training starts exactly at the pretrained model; the update is scaled by α/r; on GPT-3 175B it cut trainable parameters by about 10,000× and GPU memory by about 3×, and ranks as low as 1–4 on the attention query and value matrices worked in the original experiments.

**29. Dettmers et al.,** [**"QLoRA"**](https://arxiv.org/abs/2305.14314) **(2023).** LoRA on a 4-bit base with NF4 and paged optimisers: large models fine-tuned on one GPU (Ch 20).

**126. Schulman and Thinking Machines Lab,** [**"LoRA Without Regret"**](https://thinkingmachines.ai/blog/lora/) **(2025).** Careful comparisons of LoRA with full fine-tuning: LoRA matches full fine-tuning in many post-training settings, and for RL even at very low rank, because each episode carries little information (Ch 20). *Know cold:* the best LoRA learning rate is roughly 10× the full fine-tuning one; apply LoRA to the MLP layers, not only attention; LoRA tolerates large batches less well than full fine-tuning.

## Prompting, reasoning and agents

**30. Wei et al.,** [**"Chain-of-Thought Prompting"**](https://arxiv.org/abs/2201.11903) **(2022); Wang et al.,** [**"Self-Consistency"**](https://arxiv.org/abs/2203.11171) **(2022).** Step-by-step reasoning improves multi-step problems; sampling several chains and voting improves them further (Ch 22, 19). Self-consistency was posted in 2022 and published at ICLR 2023, so citations carry either date.

**31. ★ Yao et al.,** [**"ReAct: Synergizing Reasoning and Acting in Language Models"**](https://arxiv.org/abs/2210.03629) **(2023).** Interleave thoughts, tool actions and observations: the pattern behind most agents (Ch 33). *Know cold:* the trace interleaves Thought, Action and Observation; on knowledge tasks (HotpotQA, FEVER) the actions call a simple Wikipedia API, and reasoning-only chain of thought hallucinates facts that ReAct can look up; on decision-making benchmarks (ALFWorld, WebShop) it beat imitation and RL baselines with one or two in-context examples.

**32. Schick et al.,** [**"Toolformer"**](https://arxiv.org/abs/2302.04761) **(2023).** A model taught itself when to call APIs, keeping calls that reduced its loss (Ch 33).

**33. Khattab et al.,** [**"DSPy"**](https://arxiv.org/abs/2310.03714) **(2023).** Prompts and demonstrations as parameters optimised against a metric (Ch 22, 32).

## Inference

**123. Pope et al.,** [**"Efficiently Scaling Transformer Inference"**](https://arxiv.org/abs/2211.05102) **(2022).** The arithmetic of serving large models: how latency, throughput and cost trade off, and how to partition weights and the KV cache across chips (Ch 21). *Know cold:* decode is memory-bound at small batch because every step reads all weights, so cost per token falls with batch until compute-bound; the KV cache, not the weights, limits batch size at long context, which motivated multi-query attention; and the best partitioning differs between prefill (compute-bound) and decode (bandwidth-bound).

**124. Leviathan, Kalman and Matias,** [**"Fast Inference from Transformers via Speculative Decoding"**](https://arxiv.org/abs/2211.17192) **(2023); Chen et al.,** [**"Accelerating Large Language Model Decoding with Speculative Sampling"**](https://arxiv.org/abs/2302.01318) **(2023).** A small draft model proposes several tokens and the large model checks them in one pass, with an acceptance rule that keeps the output distribution exact (Ch 21). *Know cold:* accept draft token x with probability min(1, p(x)/q(x)), and on rejection sample from norm(max(0, p − q)); expected tokens per target pass are (1 − α^(k+1))/(1 − α) for acceptance rate α and draft length k; the gain shrinks at large batch, where the target is no longer memory-bound.

## Retrieval and RAG

**34. Robertson and Zaragoza,** [**"The Probabilistic Relevance Framework: BM25 and Beyond"**](https://www.nowpublishers.com/article/Details/INR-019) **(2009).** The theory and practice of BM25, still a strong baseline and half of most hybrid search systems (Ch 30).

**35. ★ Karpukhin et al.,** [**"Dense Passage Retrieval for Open-Domain Question Answering"**](https://arxiv.org/abs/2004.04906) **(2020).** A bi-encoder trained with in-batch negatives beat BM25 on open-domain QA (Ch 30, 29). *Know cold:* two BERT encoders scored by dot product, trained with in-batch negatives plus one BM25 hard negative per question; top-20 retrieval accuracy on Natural Questions rose from 59.1% (BM25) to 78.4%; the passage index is precomputed, which is why a dot product and not a cross-encoder.

**36. ★ Lewis et al.,** [**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**](https://arxiv.org/abs/2005.11401) **(2020).** A retriever plus a generator: grounded answers and updatable knowledge (Ch 31). *Know cold:* RAG-sequence uses the same retrieved passage for the whole output, RAG-token can switch passage per token, both marginalising over the top-k passages; the retriever is DPR and the generator BART; the query encoder is fine-tuned end to end while the document encoder and index stay fixed, so the index never needs rebuilding during training.

**37. Liu et al.,** [**"Lost in the Middle: How Language Models Use Long Contexts"**](https://arxiv.org/abs/2307.03172) **(2024).** Models use the start and end of long contexts best, with consequences for chunk ordering and context size (Ch 22, 31).

## Recommendation, ranking and experiments

**38. ★ Covington, Adams and Sargin,** [**"Deep Neural Networks for YouTube Recommendations"**](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) **(2016); Yi et al.,** [**"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"**](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) **(2019).** Candidate generation then ranking, and two-tower retrieval with in-batch negatives corrected for sampling bias: the blueprint for most large-scale recommenders, including restaurant and dish recommendation (Ch 43). *Know cold:* YouTube treats candidate generation as an extreme multiclass softmax over videos with sampled negatives, adds an "example age" feature to undo the bias towards old content, and ranks by expected watch time using weighted logistic regression; Yi et al. correct the in-batch softmax by subtracting log Q(item), the item's sampling probability estimated from a stream, because popular items appear as negatives too often.

**39. Burges,** [**"From RankNet to LambdaRank to LambdaMART: An Overview"**](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) **(2010).** Pairwise gradients scaled by the change in NDCG, plugged into boosted trees; still the strongest practical learning-to-rank baseline (Ch 44).

**40. Deng et al.,** [**"Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data"**](https://exp-platform.com/cuped/) **(CUPED, 2013).** A pre-period covariate cuts variance, so experiments need less traffic or time (Ch 2).

## Optimisation and training dynamics

**41. Yang et al.,** [**"Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"**](https://arxiv.org/abs/2203.03466) **(µP, 2022).** A parametrisation under which the best hyperparameters found on a small model transfer to a much wider one (Ch 7). *Know cold:* under the standard parametrisation the best learning rate shifts as width grows, so small-model tuning does not transfer; µP scales the output layer's multiplier and, under Adam, the hidden-matrix learning rates by 1/width so the update size per layer stays constant; the paper tuned a 40M-parameter proxy and transferred its hyperparameters to a 6.7B GPT-3-style model. It covers width, not depth.

**42. McCandlish et al.,** [**"An Empirical Model of Large-Batch Training"**](https://arxiv.org/abs/1812.06162) **(2018).** The gradient noise scale predicts the critical batch size, and it grows as training progresses (Ch 7).

**43. Liu et al.,** [**"Muon is Scalable for LLM Training"**](https://arxiv.org/abs/2502.16982) **(Moonlight, 2025).** Scaled up the Muon optimiser (orthogonalised momentum for hidden matrices) with the adjustments needed for large models (Ch 7).

**44. Micikevicius et al.,** [**"FP8 Formats for Deep Learning"**](https://arxiv.org/abs/2209.05433) **(2022).** The e4m3 and e5m2 formats and how scaling makes 8-bit training work (Ch 7).

**45. Cohen et al.,** [**"Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"**](https://arxiv.org/abs/2103.00065) **(2021).** Sharpness rises to about 2/η and hovers there; the learning rate shapes which minima training can reach (Ch 7).

**46. Wortsman et al.,** [**"Small-scale proxies for large-scale Transformer training instabilities"**](https://arxiv.org/abs/2309.14322) **(2023).** Reproduces large-run instabilities (logit growth, divergence) in small models and tests the fixes (Ch 7).

## Architectures beyond the vanilla transformer

**47. Dai et al.,** [**"DeepSeekMoE"**](https://arxiv.org/abs/2401.06066) **(2024).** Fine-grained and shared experts: the design most large MoEs now follow (Ch 11). *Know cold:* fine-grained segmentation splits each expert into m smaller ones and activates m times as many, multiplying the number of routing combinations at the same compute; shared experts that every token uses hold common knowledge, so routed experts can specialise.

**48. DeepSeek-AI,** [**"DeepSeek-V2"**](https://arxiv.org/abs/2405.04434) **(2024).** Introduced multi-head latent attention, cutting the KV cache by an order of magnitude or more (Ch 11). *Know cold:* 236B total and 21B active parameters; MLA caches a 512-dimensional latent plus a 64-dimensional decoupled RoPE key per token per layer, cutting the KV cache by 93% against DeepSeek 67B; RoPE is kept on a separate key because a position-dependent rotation would block absorbing the up-projection into the query.

**49. Gu and Dao,** [**"Mamba"**](https://arxiv.org/abs/2312.00752) **(2023); Dao and Gu,** [**"Transformers are SSMs"**](https://arxiv.org/abs/2405.21060) **(Mamba-2, 2024).** Selective state-space models with linear-time sequence mixing, and their duality with linear attention (Ch 11).

**50. Yang et al.,** [**"Gated Delta Networks"**](https://arxiv.org/abs/2412.06464) **(2024).** Combines the delta-rule update with gating for linear attention that overwrites and forgets; used in hybrid models (Ch 11).

**51. Lieber et al.,** [**"Jamba"**](https://arxiv.org/abs/2403.19887) **(2024).** An early production hybrid of Mamba, attention and MoE layers (Ch 11).

**52. Su et al.,** [**"RoFormer: Enhanced Transformer with Rotary Position Embedding"**](https://arxiv.org/abs/2104.09864) **(2021); Peng et al.,** [**"YaRN"**](https://arxiv.org/abs/2309.00071) **(2023).** RoPE, the standard positional encoding, and the most common way to extend its context (Ch 10).

**53. Gloeckle et al.,** [**"Better & Faster Large Language Models via Multi-token Prediction"**](https://arxiv.org/abs/2404.19737) **(2024).** Predicting several tokens ahead as a training signal and a built-in draft model (Ch 11).

**54. Nie et al.,** [**"Large Language Diffusion Models"**](https://arxiv.org/abs/2502.09992) **(LLaDA, 2025).** A masked diffusion LM competitive with autoregressive models of its size (Ch 11).

**55. Arora et al.,** [**"Zoology: Measuring and Improving Recall in Efficient Language Models"**](https://arxiv.org/abs/2312.04927) **(2023).** Associative recall explains most of the gap between attention and efficient alternatives (Ch 11).

## Scaling laws

**56. Besiroglu et al.,** [**"Chinchilla Scaling: A replication attempt"**](https://arxiv.org/abs/2404.10102) **(2024).** Found problems in one of Chinchilla's fits; a lesson in checking scaling-law fits (Ch 13). *Know cold:* refitting Chinchilla's third approach from data read off the paper's figures gave different constants and much wider confidence intervals than reported; the corrected fit agrees with the first two approaches on about 20 tokens per parameter. The lesson: check optimiser convergence and intervals before trusting a scaling-law fit.

**57. Sardana et al.,** [**"Beyond Chinchilla-Optimal"**](https://arxiv.org/abs/2401.00448) **(2024).** Accounting for inference makes smaller, longer-trained models optimal (Ch 13).

**58. Muennighoff et al.,** [**"Scaling Data-Constrained Language Models"**](https://arxiv.org/abs/2305.16264) **(2023).** The value of repeated data: near-fresh up to about four epochs (Ch 13, 16).

**59. Schaeffer, Miranda and Koyejo,** [**"Are Emergent Abilities of Large Language Models a Mirage?"**](https://arxiv.org/abs/2304.15004) **(2023).** Many apparent jumps come from discontinuous metrics (Ch 13).

**60. Snell et al.,** [**"Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"**](https://arxiv.org/abs/2408.03314) **(2024).** Inference compute as a scaling axis, and how to allocate it by difficulty (Ch 13, 19).

## Pretraining data

**61. Penedo et al.,** [**"The FineWeb Datasets"**](https://arxiv.org/abs/2406.17557) **(2024).** An open, documented web pipeline with ablations, and the educational-quality subset FineWeb-Edu (Ch 16). *Know cold:* 15T tokens from 96 Common Crawl snapshots; MinHash deduplication within each snapshot worked better than deduplicating across all of them, because global deduplication kept proportionally more low-quality documents; FineWeb-Edu filters with a small classifier trained on Llama-3-70B ratings of educational value, and gave large gains on knowledge and reasoning benchmarks such as MMLU and ARC.

**62. Li et al.,** [**"DataComp-LM"**](https://arxiv.org/abs/2406.11794) **(DCLM, 2024).** A benchmark for data curation; model-based filtering with a simple classifier wins (Ch 16).

**63. Lee et al.,** [**"Deduplicating Training Data Makes Language Models Better"**](https://arxiv.org/abs/2107.06499) **(2021).** Exact-substring and near-duplicate removal reduces memorisation and improves efficiency (Ch 16).

**64. Xie et al.,** [**"DoReMi"**](https://arxiv.org/abs/2305.10429) **(2023).** Learning data-mixture weights with small proxy models (Ch 16).

**65. Gunasekar et al.,** [**"Textbooks Are All You Need"**](https://arxiv.org/abs/2306.11644) **(2023); Maini et al.,** [**"Rephrasing the Web"**](https://arxiv.org/abs/2401.16380) **(2024).** Synthetic and rephrased data as a lever for small models (Ch 16).

## Training at scale

**66. Rajbhandari et al.,** [**"ZeRO"**](https://arxiv.org/abs/1910.02054) **(2020).** Sharding optimiser states, gradients and weights across data-parallel ranks (Ch 17). *Know cold:* with Ψ parameters and mixed-precision Adam (16Ψ bytes in total), stage 1 shards the 12Ψ bytes of optimiser state (4Ψ + 12Ψ/N per GPU), stage 2 also the gradients (2Ψ + 14Ψ/N), stage 3 also the weights (16Ψ/N); stages 1–2 communicate the same 2Ψ per step as plain data parallelism, stage 3 about 1.5× that (3Ψ), because weights are gathered in both passes.

**67. Shoeybi et al.,** [**"Megatron-LM"**](https://arxiv.org/abs/1909.08053) **(2019); Narayanan et al.,** [**"Efficient Large-Scale Language Model Training on GPU Clusters"**](https://arxiv.org/abs/2104.04473) **(2021).** Tensor and pipeline parallelism and how to combine them (Ch 17). *Know cold:* tensor parallelism splits the MLP's first matrix by columns and the second by rows, so each MLP and each attention block needs one all-reduce in the forward pass and one in the backward pass, which is why TP stays inside a fast NVLink domain; the 2021 paper's interleaved pipeline schedule shrinks the bubble, and combining tensor, pipeline and data parallelism reached about 52% of peak for a 1T-parameter model on 3,072 GPUs.

**68. Korthikanti et al.,** [**"Reducing Activation Recomputation in Large Transformer Models"**](https://arxiv.org/abs/2205.05198) **(2022).** Activation-memory arithmetic, sequence parallelism and selective recomputation (Ch 17).

**69. Liu, Zaharia and Abbeel,** [**"Ring Attention with Blockwise Transformers"**](https://arxiv.org/abs/2310.01889) **(2023).** Context parallelism for very long sequences (Ch 17).

## Post-training

**70. Schulman et al.,** [**"Proximal Policy Optimization Algorithms"**](https://arxiv.org/abs/1707.06347) **(2017).** The clipped objective behind RLHF (Ch 18).

**71. Gao, Schulman and Hilton,** [**"Scaling Laws for Reward Model Overoptimization"**](https://arxiv.org/abs/2210.10760) **(2022).** Gold reward rises then falls as a policy optimises a proxy; larger reward models delay it (Ch 18, 27). *Know cold:* a 6B "gold" reward model stands in for humans and proxy reward models are trained on its labels; with d = √KL from the initial policy, gold reward follows d(α − βd) for best-of-n and d(α − β log d) for RL, rising then falling; larger proxy reward models over-optimise later, and the KL penalty in RL behaved like early stopping rather than improving the frontier.

**72. Zhou et al.,** [**"LIMA: Less Is More for Alignment"**](https://arxiv.org/abs/2305.11206) **(2023).** A thousand carefully chosen SFT examples go a long way for format and style (Ch 18).

**73. Lambert et al.,** [**"Tülu 3"**](https://arxiv.org/abs/2411.15124) **(2024).** A fully open post-training recipe: SFT, on-policy preferences and RL with verifiable rewards (Ch 18, 19).

**74. Azar et al.,** [**"A General Theoretical Paradigm to Understand Learning from Human Preferences"**](https://arxiv.org/abs/2310.12036) **(IPO, 2023); Meng, Xia and Chen,** [**"SimPO"**](https://arxiv.org/abs/2405.14734) **(2024).** Two widely used DPO variants and the problems they fix (Ch 18).

**75. Huang et al.,** [**"The N+ Implementation Details of RLHF with PPO"**](https://arxiv.org/abs/2403.17031) **(2024).** The details that make or break PPO for language models (Ch 18).

## Reasoning and reinforcement learning

**76. Shao et al.,** [**"DeepSeekMath"**](https://arxiv.org/abs/2402.03300) **(2024).** Introduced GRPO, the group-relative policy optimisation behind R1-style training (Ch 19). *Know cold:* a 7B model continued on a 120B-token maths corpus mined from Common Crawl with an iteratively retrained fastText classifier; GRPO uses the group's mean and standard deviation as the baseline, puts the KL to the reference in the loss with the unbiased k3 estimator, and drops PPO's value model, which saves the memory and compute of a second policy-sized network; the paper also casts SFT, rejection sampling, DPO and PPO as one gradient form with different data sources and gradient coefficients.

**77. Lightman et al.,** [**"Let's Verify Step by Step"**](https://arxiv.org/abs/2305.20050) **(2023).** Process supervision beats outcome supervision for selecting maths solutions (Ch 19).

**78. Cobbe et al.,** [**"Training Verifiers to Solve Math Word Problems"**](https://arxiv.org/abs/2110.14168) **(2021).** Introduced GSM8K and trained a verifier to rank sampled solutions: generating many and picking with a learned verifier beat fine-tuning alone, an early form of best-of-n with an outcome reward model and of spending inference compute (Ch 19). Self-consistency, which needs no verifier, is entry 30.

**79. Yu et al.,** [**"DAPO"**](https://arxiv.org/abs/2503.14476) **(2025); Liu et al.,** [**"Understanding R1-Zero-Like Training"**](https://arxiv.org/abs/2503.20783) **(Dr. GRPO, 2025).** The biases of GRPO and practical fixes for stable large-scale RL (Ch 19).

**80. Kimi Team,** [**"Kimi k1.5"**](https://arxiv.org/abs/2501.12599) **(2025).** Scaling RL for reasoning, including long-context RL and length control (Ch 19).

**81. Muennighoff et al.,** [**"s1: Simple test-time scaling"**](https://arxiv.org/abs/2501.19393) **(2025).** A thousand examples and budget forcing give test-time scaling (Ch 19).

**82. Yue et al.,** [**"Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"**](https://arxiv.org/abs/2504.13837) **(2025).** pass@k evidence that RL mainly sharpens existing solutions (Ch 19).

## Multimodal models

**83. Dosovitskiy et al.,** [**"An Image is Worth 16x16 Words"**](https://arxiv.org/abs/2010.11929) **(ViT, 2020).** Patches as tokens (Ch 14).

**84. Zhai et al.,** [**"Sigmoid Loss for Language Image Pre-Training"**](https://arxiv.org/abs/2303.15343) **(SigLIP, 2023).** A pairwise sigmoid loss that replaced CLIP's softmax in many vision towers (Ch 14).

**85. Liu et al.,** [**"Visual Instruction Tuning"**](https://arxiv.org/abs/2304.08485) **(LLaVA, 2023); Alayrac et al.,** [**"Flamingo"**](https://arxiv.org/abs/2204.14198) **(2022).** The projector and cross-attention designs for VLMs (Ch 14).

**86. Radford et al.,** [**"Robust Speech Recognition via Large-Scale Weak Supervision"**](https://arxiv.org/abs/2212.04356) **(Whisper, 2022).** Large-scale weakly supervised speech recognition (Ch 14).

**87. Ho, Jain and Abbeel,** [**"Denoising Diffusion Probabilistic Models"**](https://arxiv.org/abs/2006.11239) **(2020); Rombach et al.,** [**"Latent Diffusion"**](https://arxiv.org/abs/2112.10752) **(2022).** The foundations of diffusion image generation (Ch 14).

**88. Chameleon Team,** [**"Chameleon"**](https://arxiv.org/abs/2405.09818) **(2024); Zhou et al.,** [**"Transfusion"**](https://arxiv.org/abs/2408.11039) **(2024).** Early-fusion and unified models that read and generate images (Ch 14).

## Evaluation

**89. Miller,** [**"Adding Error Bars to Evals"**](https://arxiv.org/abs/2411.00640) **(2024).** Standard errors, clustering and paired comparisons for model evaluation (Ch 23). *Know cold:* report the standard error √(p(1 − p)/n) with every score; cluster it when questions share a passage or template, which can widen it several-fold; compare two models with a paired test on the same items, whose variance depends on how often they disagree; and resample answers several times per question to remove the within-question sampling variance at temperature above zero.

**90. Liang et al.,** [**"Holistic Evaluation of Language Models"**](https://arxiv.org/abs/2211.09110) **(HELM, 2022).** Multi-metric, multi-scenario evaluation as a discipline (Ch 23).

**91. Rein et al.,** [**"GPQA"**](https://arxiv.org/abs/2311.12022) **(2023); Phan et al.,** [**"Humanity's Last Exam"**](https://arxiv.org/abs/2501.14249) **(2025).** Expert-written benchmarks built to resist saturation (Ch 23).

**92. Chiang et al.,** [**"Chatbot Arena"**](https://arxiv.org/abs/2403.04132) **(2024); Singh et al.,** [**"The Leaderboard Illusion"**](https://arxiv.org/abs/2504.20879) **(2025).** Pairwise human-preference leaderboards and their pitfalls (Ch 23).

**93. Jimenez et al.,** [**"SWE-bench"**](https://arxiv.org/abs/2310.06770) **(2023); Kwa et al.,** [**"Measuring AI Ability to Complete Long Tasks"**](https://arxiv.org/abs/2503.14499) **(2025).** Agentic evaluation on real software tasks, and task-length horizons as a capability trend (Ch 23, 24).

**94. Polo et al.,** [**"tinyBenchmarks"**](https://arxiv.org/abs/2402.14992) **(2024).** IRT-selected subsets cut evaluation cost with little loss (Ch 23).

**125. Chen et al.,** [**"Evaluating Large Language Models Trained on Code"**](https://arxiv.org/abs/2107.03374) **(Codex, 2021).** Introduced HumanEval, functional correctness by unit tests, and the unbiased pass@k estimator (Ch 13, 19). *Know cold:* pass@k = 1 − C(n − c, k)/C(n, k) from n ≥ k samples with c correct, because the naive 1 − (1 − ĉ)^k is biased; sampling temperature should rise with k; and BLEU-style match scores correlate poorly with whether code runs.

## Interpretability

**95. Elhage et al.,** [**"A Mathematical Framework for Transformer Circuits"**](https://transformer-circuits.pub/2021/framework/index.html) **(2021); Olsson et al.,** [**"In-context Learning and Induction Heads"**](https://arxiv.org/abs/2209.11895) **(2022).** The residual-stream view and the first well-understood circuit (Ch 26). *Know cold:* the residual stream is a shared communication channel that every layer reads from and writes to; each head splits into a QK circuit (where to attend) and an OV circuit (what to copy); induction heads, formed by composing a previous-token head with a head that uses its output, appear in a sudden phase change that coincides with a bump in the training loss and a jump in in-context learning.

**96. Meng et al.,** [**"Locating and Editing Factual Associations in GPT"**](https://arxiv.org/abs/2202.05262) **(ROME, 2022).** Causal tracing of facts to mid-layer MLPs, and rank-one edits (Ch 26).

**97. Elhage et al.,** [**"Toy Models of Superposition"**](https://arxiv.org/abs/2209.10652) **(2022).** Why neurons are polysemantic (Ch 26).

**98. Bricken et al.,** [**"Towards Monosemanticity"**](https://transformer-circuits.pub/2023/monosemantic-features) **(2023); Gao et al.,** [**"Scaling and evaluating sparse autoencoders"**](https://arxiv.org/abs/2406.04093) **(2024).** Sparse autoencoders recover interpretable features, and how to train and evaluate them at scale (Ch 26). *Know cold:* Bricken et al. trained SAEs on the 512-neuron MLP layer of a one-layer transformer with dictionaries from 512 to over 100,000 features and found most features far more interpretable than neurons; Gao et al. used TopK activations (fixing the number of active latents instead of tuning an L1 penalty), reported scaling laws for SAE size and sparsity, trained a 16M-latent SAE on GPT-4 and evaluated by the language-model loss when activations are replaced by reconstructions.

**99. Ameisen et al.,** [**"Circuit Tracing"**](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) **(2025).** Attribution graphs from transcoders trace multi-step computation in production models (Ch 26).

**100. Arditi et al.,** [**"Refusal in Language Models Is Mediated by a Single Direction"**](https://arxiv.org/abs/2406.11717) **(2024).** A striking result on linear representations and the fragility of safety training (Ch 26, 27).

## Alignment and safety

**101. Burns et al.,** [**"Weak-to-Strong Generalization"**](https://arxiv.org/abs/2312.09390) **(2023).** An empirical analogy for supervising models stronger than their supervisors (Ch 27). *Know cold:* fine-tuning GPT-4-family models on labels from a GPT-2-level supervisor recovered roughly half the performance gap on NLP tasks (PGR around 0.5), far less on chess puzzles and reward modelling; an auxiliary loss that lets the student confidently disagree with the weak labels raised PGR substantially; the analogy is imperfect because the weak model's errors are easy for the strong one to imitate.

**102. Sharma et al.,** [**"Towards Understanding Sycophancy in Language Models"**](https://arxiv.org/abs/2310.13548) **(2023).** Sycophancy across RLHF-trained assistants and its roots in preference data (Ch 27).

**103. Zou et al.,** [**"Universal and Transferable Adversarial Attacks on Aligned Language Models"**](https://arxiv.org/abs/2307.15043) **(GCG, 2023).** Optimisation-based jailbreak suffixes that transfer across models (Ch 27).

**104. Hubinger et al.,** [**"Sleeper Agents"**](https://arxiv.org/abs/2401.05566) **(2024); Greenblatt et al.,** [**"Alignment Faking in Large Language Models"**](https://arxiv.org/abs/2412.14093) **(2024).** Deceptive behaviours that survive or game safety training, in constructed settings (Ch 27).

**105. Betley et al.,** [**"Emergent Misalignment"**](https://arxiv.org/abs/2502.17424) **(2025).** Narrow fine-tuning producing broad misalignment (Ch 27).

**106. Greenblatt et al.,** [**"AI Control"**](https://arxiv.org/abs/2312.06942) **(2023); Baker et al.,** [**"Monitoring Reasoning Models for Misbehavior"**](https://arxiv.org/abs/2503.11926) **(2025).** Safety without verified alignment, and chain-of-thought monitoring with its fragility (Ch 27).

## Open model and system reports

**107. Llama Team,** [**"The Llama 3 Herd of Models"**](https://arxiv.org/abs/2407.21783) **(2024).** The most detailed account of a dense frontier-scale pretraining run, including infrastructure and failures (Ch 17, 16, 15). *Know cold:* the 405B dense model trained on about 15.6T tokens with about 3.8e25 FLOPs on up to 16k H100s; the team logged 466 job interruptions in a 54-day snapshot, most from hardware such as GPUs and HBM; post-training used SFT, rejection sampling and DPO rather than PPO, and context was extended to 128k tokens in stages late in pretraining.

**108. DeepSeek-AI,** [**"DeepSeek-V3 Technical Report"**](https://arxiv.org/abs/2412.19437) **(2024).** MoE with MLA, auxiliary-loss-free balancing, multi-token prediction and fp8 training at scale (Ch 11, 17). *Know cold:* 671B total and 37B active parameters, trained on 14.8T tokens for about 2.8M H800 GPU-hours; the first openly documented fp8 training at this scale, with fine-grained (tile- and block-wise) scaling; auxiliary-loss-free balancing by per-expert biases, a multi-token-prediction module that can serve as a speculative draft, and the DualPipe schedule that overlaps all-to-all communication with compute.

**109. OLMo Team,** [**"2 OLMo 2 Furious"**](https://arxiv.org/abs/2501.00656) **(2024).** A fully open model with data, code and checkpoints; annealing and mid-training documented (Ch 16).

**110. Qwen Team,** [**"Qwen3 Technical Report"**](https://arxiv.org/abs/2505.09388) **(2025).** Dense and MoE families with thinking and non-thinking modes in one model and a documented post-training pipeline; read at least one new open technical report a month to stay current (Ch 12, 11, 19).

## Recent work to the end of 2025: RL at scale and efficient architectures

**111. Ahmadian et al.,** [**"Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback"**](https://arxiv.org/abs/2402.14740) **(RLOO, 2024).** REINFORCE with a leave-one-out baseline matches PPO for RLHF without a value model; the clearest bridge from textbook policy gradients to GRPO-style methods (Ch 18, 19).

**112. Zheng et al.,** [**"Group Sequence Policy Optimization"**](https://arxiv.org/abs/2507.18071) **(GSPO, 2025); MiniMax,** [**"MiniMax-M1"**](https://arxiv.org/abs/2506.13585) **(CISPO, 2025).** Two answers to unstable token-level importance ratios: a length-normalised sequence ratio, and clipping the weight while keeping every token's gradient (Ch 19).

**113. Yao et al.,** [**"Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"**](https://fengyao.notion.site/off-policy-rl) **(2025); He et al.,** [**"Defeating Nondeterminism in LLM Inference"**](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) **(2025).** The rollout engine and trainer disagree on token probabilities; truncated importance sampling corrects for it, and batch-invariant kernels remove it (Ch 19).

**114. Khatri et al.,** [**"The Art of Scaling Reinforcement Learning Compute for LLMs"**](https://arxiv.org/abs/2510.13786) **(ScaleRL, 2025).** Sigmoid compute–reward curves for RL; most recipe details change efficiency rather than the ceiling (Ch 19).

**115. Gunjal et al.,** [**"Rubrics as Rewards"**](https://arxiv.org/abs/2507.17746) **(2025).** Per-prompt weighted rubrics scored by a judge extend RLVR to tasks without a single correct answer (Ch 19).

**116. Lu et al.,** [**"On-Policy Distillation"**](https://thinkingmachines.ai/blog/on-policy-distillation/) **(2025); Agarwal et al.,** [**"On-Policy Distillation of Language Models"**](https://arxiv.org/abs/2306.13649) **(GKD, 2023).** The student samples, the teacher grades every token: RL's on-policy data with SFT's dense signal (Ch 15, 19).

**117. Kimi Team,** [**"Kimi K2: Open Agentic Intelligence"**](https://arxiv.org/abs/2507.20534) **(2025).** A 1T-parameter MoE trained with Muon plus QK-Clip on 15.5T tokens without loss spikes, and a large agentic data pipeline (Ch 7, 12).

**118. Yuan et al.,** [**"Native Sparse Attention"**](https://arxiv.org/abs/2502.11089) **(2025); DeepSeek-AI,** [**"DeepSeek-V3.2"**](https://arxiv.org/abs/2512.02556) **(2025).** Trainable sparse attention designed with its kernels, and a recipe for retrofitting it onto a dense model with a distilled indexer (Ch 11).

**119. Xie et al.,** [**"mHC: Manifold-Constrained Hyper-Connections"**](https://arxiv.org/abs/2512.24880) **(2025).** Wider residual streams with mixing matrices constrained to be doubly stochastic, restoring the identity path that deep networks need (Ch 7).

This section collects 2024–25 work that changed practice: two entries (111 and the GKD half of 116) predate 2025 but belong with the methods they enabled. No 2026 papers are listed: the edition stops at the end of 2025 rather than include results that have not yet been replicated or built on. Entries 113, 114 and the reports in 107–110 together are the best picture, as of that date, of how open labs run RL. For anything newer, check the latest open technical reports from DeepSeek, Qwen, Kimi, MiniMax and AI2 (OLMo, Tülu) first, and apply the same test as for this list: does the result change what a practitioner would do, and has anyone else reproduced it?

## Also worth a look

Sculley et al., ["Hidden Technical Debt in Machine Learning Systems"](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/) (2015), for MLOps (Ch 47) · Holtzman et al., ["The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751) (2020), for nucleus sampling (Ch 21) · Kwon et al., ["PagedAttention / vLLM"](https://arxiv.org/abs/2309.06180) (2023), for serving (Ch 21) · Greshake et al., [indirect prompt injection](https://arxiv.org/abs/2302.12173) (2023), for safety (Ch 28) · Zheng et al., ["Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) (2023), for evaluation (Ch 24).

## A reading plan

- **Week 1:** the ten starred papers (abstract, figures, method, conclusion), and their *Know cold* notes said aloud.
- **Week 2:** the papers closest to your own work or target role, for example 27–29 (fine-tuning), 22 and 24 (alignment and preference data), 35–36 (RAG) or 38–39 (recommendations).
- **Research track (instead of, or after, week 2):** the research core above over two weeks, then one area per week from 41–126.
- **Ongoing:** one paper every few days from the rest, with a three-sentence summary in your own words: problem, idea, why it matters for the products you work on.

---

[← C · Notation and common formulas](C-notation-and-formulas.md) · [Contents](../README.md#contents) · [E · Worked system designs →](E-worked-designs.md)
