# D · Paper map

Forty papers that come up again and again in applied ML and LLM work and interviews, grouped by theme, one paragraph each: the problem it solved, the key idea, and why it still matters. Every title links to the paper. Read the abstract, the figures and the conclusion first; go deeper where your role needs it. A star (★) marks ten to know cold for any applied LLM role.

## Foundations of deep learning

**1. Rumelhart, Hinton and Williams,** [**"Learning representations by back-propagating errors"**](https://www.nature.com/articles/323533a0) **(1986).** Showed that the chain rule, applied layer by layer, trains multi-layer networks that learn useful internal features. Every model in this guide is trained this way (Ch 6).

**2. Kingma and Ba,** [**"Adam: A Method for Stochastic Optimization"**](https://arxiv.org/abs/1412.6980) **(2015); Loshchilov andHutter,** [**"Decoupled Weight Decay Regularization"**](https://arxiv.org/abs/1711.05101) **(AdamW, 2019).** Momentum plus per-parameter step sizes with bias correction; AdamW separates weight decay and is the default for transformers (Ch 6).

**3. Srivastava et al.,** [**"Dropout"**](https://jmlr.org/papers/v15/srivastava14a.html) **(2014).** Randomly zeroing units during training stops co-adaptation and acts like averaging many thinned networks (Ch 6).

**4. Ioffe and Szegedy,** [**"Batch Normalization"**](https://arxiv.org/abs/1502.03167) **(2015); Ba, Kiros and Hinton,** [**"Layer Normalization"**](https://arxiv.org/abs/1607.06450) **(2016).** Normalising activations made deep networks train faster and more stably; transformers use layer norm because it does not depend on the batch (Ch 6, 10).

**5. He et al.,** [**"Deep Residual Learning for Image Recognition"**](https://arxiv.org/abs/1512.03385) **(2016).** Skip connections let gradients flow through very deep networks. The residual stream is the backbone of the transformer (Ch 6, 10).

## Representations and sequence models

**6. Mikolov et al.,** [**"Distributed Representations of Words and Phrases and their Compositionality"**](https://arxiv.org/abs/1310.4546) **(2013).** Skip-gram with negative sampling made word embeddings cheap; the same trick embeds items from baskets and sessions (Ch 8, 25).

**7. Hochreiter and Schmidhuber,** [**"Long Short-Term Memory"**](https://doi.org/10.1162/neco.1997.9.8.1735) **(1997).** A gated cell updated additively lets gradients survive long sequences (Ch 8).

**8. Bahdanau, Cho and Bengio,** [**"Neural Machine Translation by Jointly Learning to Align and Translate"**](https://arxiv.org/abs/1409.0473) **(2014).** Attention removed the single-vector bottleneck of seq2seq: the direct ancestor of the transformer (Ch 8).

**9. Sennrich, Haddow and Birch,** [**"Neural Machine Translation of Rare Words with Subword Units"**](https://arxiv.org/abs/1508.07909) **(2016).** Brought byte-pair encoding to NLP; every modern tokenizer descends from it (Ch 9).

## Transformers and pre-trained models

**10. ★ Vaswani et al.,** [**"Attention Is All You Need"**](https://arxiv.org/abs/1706.03762) **(2017).** Replaced recurrence with multi-head self-attention, positional encodings, residuals and feed-forward blocks, making training parallel. Be able to implement the block and justify each design choice (Ch 10).

**11. ★ Devlin et al.,** [**"BERT"**](https://arxiv.org/abs/1810.04805) **(2019).** A bidirectional encoder pre-trained with masked language modelling, then fine-tuned; still the right family for high-volume classification and reranking (Ch 11, 12).

**12. Radford et al.,** [**"Language Models are Unsupervised Multitask Learners"**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) **(GPT-2,2019) and Brown et al.,** [**"Language Models are Few-Shot Learners"**](https://arxiv.org/abs/2005.14165) **(GPT-3, 2020).** Scaling a decoder-only model produced zero- and few-shot ability from prompts alone (Ch 11, 15).

**13. Raffel et al.,** [**"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**](https://arxiv.org/abs/1910.10683) **(T5, 2020).** Every task as text-to-text, plus a model study of objectives and data (Ch 11).

**14. Reimers and Gurevych,** [**"Sentence-BERT"**](https://arxiv.org/abs/1908.10084) **(2019).** Siamese fine-tuning made encoder embeddings work for semantic search at scale (Ch 11, 17).

**15. Radford et al.,** [**"Learning Transferable Visual Models From Natural Language Supervision"**](https://arxiv.org/abs/2103.00020) **(CLIP, 2021).** Contrastive training on image–caption pairs created a shared image–text space with zero-shot classification (Ch 11).

## Scaling and efficient models

**16. Kaplan et al.,** [**"Scaling Laws for Neural Language Models"**](https://arxiv.org/abs/2001.08361) **(2020).** Loss follows smooth power laws in parameters, data and compute (Ch 11).

**17. ★ Hoffmann et al.,** [**"Training Compute-Optimal Large Language Models"**](https://arxiv.org/abs/2203.15556) **(Chinchilla,2022).** For fixed compute, scale parameters and tokens equally, about 20 tokens per parameter; know C ≈ 6ND and why practice now overtrains small models (Ch 11).

**18. Fedus, Zoph and Shazeer,** [**"Switch Transformers"**](https://arxiv.org/abs/2101.03961) **(2021).** Top-1 mixture-of-experts routing with a load-balancing loss: more parameters, same compute per token (Ch 11).

**19. Dao et al.,** [**"FlashAttention"**](https://arxiv.org/abs/2205.14135) **(2022).** Exact attention computed in tiles to cut memory traffic; hardware-aware engineering as a source of speed (Ch 10).

**20. Ainslie et al.,** [**"GQA"**](https://arxiv.org/abs/2305.13245) **(2023).** Shared key-value heads shrink the KV cache with little quality loss (Ch 10, 14).

## Training LLMs to follow instructions

**21. Wei et al.,** [**"Finetuned Language Models Are Zero-Shot Learners"**](https://arxiv.org/abs/2109.01652) **(FLAN, 2022).** Instruction tuning on many tasks improved zero-shot performance on unseen tasks (Ch 13).

**22. ★ Ouyang et al.,** [**"Training language models to follow instructions with human feedback"**](https://arxiv.org/abs/2203.02155) **(InstructGPT, 2022).** The SFT → reward model → PPO recipe with a KL penalty (Ch 13).

**23. Bai et al.,** [**"Constitutional AI: Harmlessness from AI Feedback"**](https://arxiv.org/abs/2212.08073) **(2022).** Self-critique against written principles, and AI-labelled preferences; the basis of RLAIF and LLM-as-judge thinking (Ch 13, 22).

**24. ★ Rafailov et al.,** [**"Direct Preference Optimization"**](https://arxiv.org/abs/2305.18290) **(2023).** KL-constrained RLHF rewritten as a classification loss on preference pairs; no reward model, no RL loop (Ch 13).

**25. Hinton, Vinyals and Dean,** [**"Distilling the Knowledge in a Neural Network"**](https://arxiv.org/abs/1503.02531) **(2015).** A student trained on a teacher's softened outputs; the route from an expensive model to a cheap one (Ch 13).

**26. DeepSeek-AI,** [**"DeepSeek-R1"**](https://arxiv.org/abs/2501.12948) **(2025), with GRPO from Shao et al.,** [**"DeepSeekMath"**](https://arxiv.org/abs/2402.03300) **(2024).** RL with verifiable rewards produced long reasoning; GRPO normalises rewards within groups instead of using a value model (Ch 13).

## Fine-tuning efficiently

**27. Li and Liang,** [**"Prefix-Tuning"**](https://arxiv.org/abs/2101.00190) **(2021); Lester, Al-Rfou and Constant,** [**"The Power of Scale for Parameter-Efficient Prompt Tuning"**](https://arxiv.org/abs/2104.08691) **(2021).** Train only continuous prompt vectors on a frozen model; prompt tuning approaches full fine-tuning as models grow (Ch 16).

**28. ★ Hu et al.,** [**"LoRA: Low-Rank Adaptation of Large Language Models"**](https://arxiv.org/abs/2106.09685) **(2021).** Low-rank updates added to frozen weights; tiny trainable parameter count and no added latency once merged (Ch 16).

**29. Dettmers et al.,** [**"QLoRA"**](https://arxiv.org/abs/2305.14314) **(2023).** LoRA on a 4-bit base with NF4 and paged optimisers: large models fine-tuned on one GPU (Ch 16).

## Prompting, reasoning and agents

**30. Wei et al.,** [**"Chain-of-Thought Prompting"**](https://arxiv.org/abs/2201.11903) **(2022); Wang et al.,** [**"Self-Consistency"**](https://arxiv.org/abs/2203.11171) **(2023).** Step-by-step reasoning improves multi-step problems; sampling several chains and voting improves them further (Ch 15).

**31. ★ Yao et al.,** [**"ReAct: Synergizing Reasoning and Acting in Language Models"**](https://arxiv.org/abs/2210.03629) **(2023).** Interleave thoughts, tool actions and observations: the pattern behind most agents (Ch 21).

**32. Schick et al.,** [**"Toolformer"**](https://arxiv.org/abs/2302.04761) **(2023).** A model taught itself when to call APIs, keeping calls that reduced its loss (Ch 21).

**33. Khattab et al.,** [**"DSPy"**](https://arxiv.org/abs/2310.03714) **(2023).** Prompts and demonstrations as parameters optimised against a metric (Ch 15, 20).

## Retrieval and RAG

**34. Robertson and Zaragoza,** [**"The Probabilistic Relevance Framework: BM25 and Beyond"**](https://www.nowpublishers.com/article/Details/INR-019) **(2009).** The theory and practice of BM25, still a strong baseline and half of most hybrid search systems (Ch 18).

**35. ★ Karpukhin et al.,** [**"Dense Passage Retrieval for Open-Domain Question Answering"**](https://arxiv.org/abs/2004.04906) **(2020).** A bi-encoder trained with in-batch negatives beat BM25 on open-domain QA (Ch 17, 18).

**36. ★ Lewis et al.,** [**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**](https://arxiv.org/abs/2005.11401) **(2020).** A retriever plus a generator: grounded answers and updatable knowledge (Ch 19).

**37. Liu et al.,** [**"Lost in the Middle: How Language Models Use Long Contexts"**](https://arxiv.org/abs/2307.03172) **(2024).** Models use the start and end of long contexts best, with consequences for chunk ordering and context size (Ch 15, 19).

## Recommendation, ranking and experiments

**38. ★ Covington, Adams and Sargin,** [**"Deep Neural Networks for YouTube Recommendations"**](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) **(2016); Yi et al.,** [**"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"**](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) **(2019).** Candidate generation then ranking, and two-tower retrieval with in-batch negatives corrected for sampling bias: the blueprint for most large-scale recommenders, including restaurant and dish recommendation (Ch 25).

**39. Burges,** [**"From RankNet to LambdaRank to LambdaMART: An Overview"**](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) **(2010).** Pairwise gradients scaled by the change in NDCG, plugged into boosted trees; still the strongest practical learning-to-rank baseline (Ch 26).

**40. Deng et al.,** [**"Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data"**](https://exp-platform.com/cuped/) **(CUPED, 2013).** A pre-period covariate cuts variance, so experiments need less traffic or time (Ch 2).

## Also worth a look

Sculley et al., ["Hidden Technical Debt in Machine Learning Systems"](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/) (2015), for MLOps (Ch 29) · Holtzman et al., ["The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751) (2020), for nucleus sampling (Ch 14) · Kwon et al., ["PagedAttention / vLLM"](https://arxiv.org/abs/2309.06180) (2023), for serving (Ch 14) · Greshake et al., [indirect prompt injection](https://arxiv.org/abs/2302.12173) (2023), for safety (Ch 24) · Zheng et al., ["Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) (2023), for evaluation (Ch 22).

## A reading plan

- **Week 1:** the ten starred papers (abstract, figures, method, conclusion).
- **Week 2:** the papers closest to your own work or target role, for example 27–29 (fine-tuning), 22 and 24 (alignment and preference data), 35–36 (RAG) or 38–39 (recommendations).
- **Ongoing:** one paper every few days from the rest, with a three-sentence summary in your own words: problem, idea, why it matters for the products you work on.

---

[← C · Notation and formulas](C-notation-and-formulas.md) · [Contents](../README.md#contents) · [E · Worked designs →](E-worked-designs.md)
