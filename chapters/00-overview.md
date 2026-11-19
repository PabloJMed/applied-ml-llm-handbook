# Overview

A self-study handbook for AI and LLM research scientists, and for the applied ML engineers who work alongside them: from the maths and statistics underneath, through how large language models are pretrained, post-trained, served, evaluated and understood, to the research craft and interviews of the job. Theory, working code and worked numbers in one place.

**All chapters:** see the [linked contents in the README](../README.md#contents).

## How the handbook works

The handbook has 48 chapters in 11 parts and six appendices. Parts I–X are the core path for a research scientist working on LLMs. Part XI is an optional applied track (classic models, recommender and search systems, ML system design, MLOps and data engineering) for applied roles and for anyone whose research meets a product.

**Every chapter follows the same template:**

1. **Why it matters.** Where this shows up in research, on the job and in interviews.
2. **Key concepts.** The concepts, models and papers of the chapter, each explained in a line with a link to go deeper.
3. **Core sections.** Intuition first, then the maths, then working code (Python, NumPy, PyTorch). Numbers in the text come from running the chapter's script in `code/`.
4. **Pitfalls.** The mistakes practitioners make and reviewers and interviewers probe.
5. **Interview questions** with model answers, from basic to senior.
6. **Practice**, including a **research exercise**: reproduce a result, run an ablation, or measure variance, rather than only drill.
7. **Read next.** Papers, docs and books.
8. **Done when.** What you should be able to explain or build without notes.

**How to use it:** read a section, close it, explain it aloud, then do the practice. Run the chapter's script and change something. If you use an AI assistant, ask it to review your explanation or code against the chapter. Code is written for understanding first; it runs, but it is not production-hardened.

**A caution on a fast-moving field:** methods and tools change every few months. The book states dates where they matter, and concepts outlast APIs; check library calls against current documentation and recent results against newer papers.

## Chapter guide

### Part I — Foundations

| Ch | Title | What it covers |
|---|---|---|
| 1 | The maths you actually use | Vectors, SVD, probability, MLE and MAP, gradients, matrix calculus and curvature, entropy and KL (forward vs reverse), perplexity, compression as prediction |
| 2 | Statistics and experimentation | Confidence intervals, tests, power, A/B tests, CUPED, multiple testing, causal basics, statistics for model evaluation |
| 3 | ML fundamentals | Framing, bias–variance, losses, regularisation, optimisation basics, splits, leakage, calibration |
| 4 | Evaluation and metrics | Classification, ranking and regression metrics, calibration, slicing, confidence intervals on metrics |

### Part II — Deep learning and training dynamics

| Ch | Title | What it covers |
|---|---|---|
| 5 | Neural network fundamentals | MLPs, backpropagation by hand, activations, initialisation, normalisation, debugging |
| 6 | PyTorch for practitioners | Tensors and autograd, modules, the training loop, mixed precision, checkpointing, reproducibility |
| 7 | Optimisation and training dynamics | AdamW, schedules (cosine, WSD), batch size and gradient noise, Muon and other optimisers, µP, bf16/fp8, loss spikes, z-loss, QK-norm, reading training curves |
| 8 | Embeddings and sequence models | word2vec, RNNs, LSTMs, seq2seq and the first attention; the link to modern recurrent models |

### Part III — Transformers and architectures

| Ch | Title | What it covers |
|---|---|---|
| 9 | Text and tokenisation | BPE from scratch, tokenizer trade-offs, multilingual cost, tokenizer research (vocabulary scaling, byte-level models) |
| 10 | The transformer in depth | Attention, GQA, RoPE, normalisation, SwiGLU, the KV cache, a mini-GPT; long-context extension (PI, NTK, YaRN), attention sinks |
| 11 | Architectures beyond the vanilla transformer | Mixture of experts and load balancing, MLA, sparse and sliding attention, linear attention and SSMs, hybrids, multi-token prediction, diffusion LMs |
| 12 | Model families and scaling | Encoders, decoders and encoder–decoders; sentence transformers; MoE and scaling in overview; choosing a model |
| 13 | Scaling laws | Kaplan vs Chinchilla, deriving the compute-optimal allocation, IsoFLOP fits, over-training for inference, repeated data, emergence, test-time compute |
| 14 | Multimodal models | ViT, CLIP and SigLIP, VLM architectures and training, resolution and token budgets, speech, image generation, video, multimodal evaluation |

### Part IV — Pretraining

| Ch | Title | What it covers |
|---|---|---|
| 15 | How LLMs are trained | The pipeline end to end: pretraining, SFT, RLHF, DPO, RLVR and distillation, and what each stage explains |
| 16 | Pretraining data | Web pipelines, heuristic and model-based filtering, MinHash deduplication, decontamination, mixtures, annealing and mid-training, synthetic data, data ablations |
| 17 | Pretraining at scale | FLOPs, time and MFU; memory anatomy; ring all-reduce, ZeRO and FSDP; tensor, pipeline, context and expert parallelism; checkpointing and failures |

### Part V — Post-training

| Ch | Title | What it covers |
|---|---|---|
| 18 | Post-training I: SFT and preferences | SFT data, masking and packing, reward models, policy-gradient foundations (REINFORCE, baselines, importance sampling), PPO with GAE, KL estimators, over-optimisation, DPO derivation and variants, AI feedback |
| 19 | Post-training II: RL and reasoning | RL with verifiable rewards, GRPO and its fixes (Dr. GRPO, DAPO), verifiers and reward hacking, curricula, does RL add reasoning, PRMs, test-time strategies, distilling reasoning, RL systems |
| 20 | Fine-tuning and parameter-efficient methods | When to fine-tune, LoRA and QLoRA in depth, adapters, instruction data, forgetting, continued pretraining, model merging, RL fine-tuning |

### Part VI — Inference and prompting

| Ch | Title | What it covers |
|---|---|---|
| 21 | Inference, decoding and serving | Decoding strategies, the KV cache, prefill and decode, serving engines, speculative decoding, quantisation, cost arithmetic, roofline, disaggregation, serving MoE and reasoning models |
| 22 | Prompting and context engineering | Prompt anatomy, few-shot, reasoning, structured output, long context, caching, prompt injection, in-context learning research |

### Part VII — Evaluation science

| Ch | Title | What it covers |
|---|---|---|
| 23 | Evaluating models | Error bars, clustering and paired tests, sample size, sources of variance, bits per byte, benchmark design and IRT, contamination, arenas, judge biases, agent horizons |
| 24 | LLM evaluation | Evaluating LLM applications: eval sets, task metrics, LLM judges and their calibration, regression testing |
| 25 | RAG and agent evaluation | Retrieval and faithfulness metrics, end-state and trajectory evaluation, pass^k reliability |

### Part VIII — Interpretability, alignment and safety

| Ch | Title | What it covers |
|---|---|---|
| 26 | Interpretability and analysis | Residual stream view, probing, logit lens, activation patching, induction circuits, superposition and sparse autoencoders, circuit tracing, steering |
| 27 | Alignment and safety research | Reward hacking, sycophancy, jailbreaks and defences, automated red-teaming, scalable oversight and weak-to-strong, deception evaluations, control and CoT monitoring, dangerous-capability evaluations |
| 28 | Safety and responsible AI | Responsible AI in products: moderation, privacy, fairness, injection defences, regulation |

### Part IX — Retrieval and agents

| Ch | Title | What it covers |
|---|---|---|
| 29 | Embeddings and representation learning | Contrastive learning, hard negatives, fine-tuning embedders, Matryoshka, quantisation, LLM-based embedders |
| 30 | Information retrieval | BM25, dense retrieval, ANN indexes, filtering, hybrid search, re-ranking, query rewriting, retrieval evaluation, learned sparse and reasoning-intensive retrieval |
| 31 | Retrieval-augmented generation | Chunking, retrieval pipelines, citations, failure modes and fixes |
| 32 | Frameworks and tooling | Application frameworks (LangChain, LangGraph, LlamaIndex, DSPy) and the research stack (Megatron, torchtitan, TRL, verl, vLLM, eval harnesses) |
| 33 | Agents and tool use | Workflows vs agents, function calling, planning, memory, MCP, guardrails, reliability; training agents with RL |

### Part X — Research craft and interviews

| Ch | Title | What it covers |
|---|---|---|
| 34 | Research craft | Choosing problems, reading and reproducing papers, experiment design and ablations, seeds and statistics, compute budgeting, writing research memos |
| 35 | The ML coding interview | pandas and SQL patterns, classic algorithms, and from-scratch components for research rounds (attention, losses, samplers, routers) |
| 36 | Research interviews | The research-design round with worked answers, paper discussions, research talks, take-home tasks |
| 37 | ML depth question bank | 250 questions with short answers, including 90 on LLM research and RL; follow-up chains and graded answers |
| 38 | Behavioural interviews | STAR, a story bank, senior signals, research stories |
| 39 | Thinking like a senior | Trade-offs, scoping, uncertainty, influence, research judgement |

### Part XI — Applied ML systems (optional track)

| Ch | Title | What it covers |
|---|---|---|
| 40 | Classic models | Linear and logistic regression, trees, forests, boosting, kNN, SVMs, naive Bayes, k-means, PCA, from scratch |
| 41 | NLP tasks with modern tools | Classification with LLMs and distillation, NER, sentiment, topics, summarisation, translation, task metrics |
| 42 | Infrastructure primer | Servers, GPUs, APIs, Docker, Kubernetes, queues, batch vs online inference, capacity planning |
| 43 | Recommender systems | Collaborative filtering, two-tower retrieval, ranking, bandits, position bias, off-policy evaluation, generative recommenders |
| 44 | Search and learning to rank | Query understanding, BM25, LTR (RankNet to LambdaMART), fusion, LLM rerankers |
| 45 | Marketplace and logistics ML | Forecasting, ETA, matching, pricing |
| 46 | ML system design | The eight-step method, back-of-the-envelope maths, LLM systems, research system design |
| 47 | MLOps and LLMOps | Pipelines, registries, release ladders, monitoring, LLMOps, research-scale operations |
| 48 | Data for ML | SQL patterns, point-in-time joins, Spark, data contracts, labelling |

### Appendices

- **A.** Glossary
- **B.** One-page cheat sheets, including LLM research
- **C.** Notation and formulas
- **D.** Paper map: the papers worth knowing, from foundations to the end of 2025
- **E.** Ten worked system designs, the model answers for Chapter 46
- **F.** A 40-question diagnostic with an answer key: take it first, and again after four weeks

## Study plans

**Start with the diagnostic.** Appendix F has 40 questions, five per area, with an answer key and a rule for turning the scores into a plan. Use the plans below as the default order, then spend extra time on the areas where you scored 0–4 and skip ahead where you scored 8–10.

**Research track: twelve weeks to LLM research-scientist readiness.** Assumes you know ML and some deep learning. About four chapters a week, with the research exercises as the main practice.

| Week | Chapters | Focus |
|---|---|---|
| 1 | 1, 2, 5, 6 | Refresh the maths, statistics and deep-learning mechanics |
| 2 | 7, 9, 10 | Optimisation and the transformer in depth |
| 3 | 11, 12, 13 | Architectures and scaling laws |
| 4 | 15, 16, 17 | Pretraining: data and systems |
| 5 | 18, 19 | Post-training and reasoning (the densest week) |
| 6 | 20, 21, 22 | Fine-tuning, inference and prompting |
| 7 | 23, 24, 25 | Evaluation science |
| 8 | 26, 27 | Interpretability and alignment |
| 9 | 14, 29, 30, 33 | Multimodal, retrieval and agents |
| 10 | 34, 36 | Research craft and research interviews; start a small research project |
| 11 | 35, 37 | Coding components and the depth bank, timed |
| 12 | 38, 39, Appendices B and D | Stories, talk rehearsal, final review |

**Applied track: eight weeks for applied ML and LLM engineering roles.**

| Week | Chapters | Focus |
|---|---|---|
| 1 | 46, 43 | The design framework and recommender systems; start weekly timed design practice |
| 2 | 4, 44 | Metrics and search ranking |
| 3 | 30, 41, 22 | Retrieval, LLM classification, prompting |
| 4 | 31, 24, 25 | RAG and its evaluation |
| 5 | 3, 40, 10 | ML fundamentals and the transformer |
| 6 | 29, 20, 33 | Embeddings, fine-tuning, agents |
| 7 | 2, 45, 38 | Experimentation, marketplace ML, behavioural stories |
| 8 | 35, 37, Appendix B | The mock loop |

**Two-week refresh** (you have done the work before and the interview is close). Week 1: take the diagnostic, then each day one red or amber area: its chapters' pitfalls, interview questions and "Done when" box, plus the matching section of the question bank aloud. Week 2: one timed design (Appendix E) and one from-scratch coding component (Chapter 35.9) each day, the follow-up chains in Chapter 37, and your story bank (Chapter 38) rehearsed aloud.

**Seventy-two hours.** Day 1: the cheat sheets (Appendix B) and the question bank's senior questions in your target area, aloud and timed. Day 2: two timed system designs and two coding components; the follow-up chains. Day 3: five stories in STAR form, your one-minute introduction (Chapter 39.7), questions to ask your interviewers (Chapter 36.7), and sleep.

## Core library

Most of these are free. Each chapter's "Read next" points to specific chapters or sections.

### Books and long-form guides

| Title | Used in chapters | Cost |
|---|---|---|
| [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/), Simon Prince (2023) | 1, 5, 7, 10, 14 | Free PDF |
| [*Dive into Deep Learning*](https://d2l.ai/), Zhang et al. | 5, 6, 8, 10 | Free online |
| [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/), Google DeepMind (2025) | 13, 17, 21 | Free online |
| [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook), Hugging Face (2025) | 17 | Free online |
| [*RLHF Book*](https://rlhfbook.com/), Nathan Lambert | 18, 19 | Free online |
| [*Speech and Language Processing* (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/), Jurafsky and Martin | 8, 9, 10, 30 | Free online |
| *Build a Large Language Model (From Scratch)*, Sebastian Raschka | 9, 10, 18 | Optional |
| *AI Engineering*, Chip Huyen (2024) | 21, 22, 24, 31, 33 | Optional, applied track |
| *Machine Learning System Design Interview*, Aminian and Xu | 43, 44, 46 | Optional, applied track |
| [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/), Manning, Raghavan, Schütze | 30, 44 | Free online |
| *Trustworthy Online Controlled Experiments*, Kohavi, Tang, Xu | 2 | Optional |

### Blogs, courses and research feeds

- [Andrej Karpathy, Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — chapters 5, 9, 10
- [Lilian Weng's blog](https://lilianweng.github.io/) — chapters 15, 18, 19, 27, 33
- [Transformer Circuits Thread](https://transformer-circuits.pub/) — chapter 26
- [Sebastian Raschka's Ahead of AI](https://magazine.sebastianraschka.com/) — chapters 11, 18, 19
- [Nathan Lambert's Interconnects](https://www.interconnects.ai/) — chapters 18, 19
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) — chapters 9, 13, 16, 17
- [Eugene Yan's writing](https://eugeneyan.com/) and [Hamel Husain's blog](https://hamel.dev/) — chapters 24, 25, 43, 46

### Papers worth reading in full

| Paper | Chapter |
|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) | 10 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020) | 13 |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) (2022) | 13 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) | 18 |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (2023) | 18 |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023) | 19 |
| [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) (2024) | 16 |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024) | 15, 17 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024) | 11, 17 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025) | 19 |
| [Tülu 3](https://arxiv.org/abs/2411.15124) (2024) | 18 |
| [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (2022) | 26 |
| [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) (2022) | 26 |
| [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (2024) | 23 |
| [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390) (2023) | 27 |
| [LoRA](https://arxiv.org/abs/2106.09685) (2021) | 20 |
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (2020) | 31 |
| [ReAct](https://arxiv.org/abs/2210.03629) (2022) | 33 |

Appendix D maps more than a hundred papers by area. The arXiv IDs are the papers' standard identifiers.

---

[Contents](../README.md#contents) · [1 · The maths you actually use →](01-the-maths-you-actually-use.md)
