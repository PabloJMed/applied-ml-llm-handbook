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

**A caution on a fast-moving field:** methods and tools change every few months. The book states dates where they matter, and concepts outlast APIs; check library calls against current documentation and recent results against newer papers. The foundations (Parts I and II, most of Part III) date slowly. The most perishable material is in Chapters 17 (training hardware and parallelism), 19 (RL for reasoning), 21 (serving), 22 (prompting and context engineering), 23 (benchmarks), 33 (agents) and Appendix D (the paper map): before an interview, spend an hour on what has been published since in those areas, starting from the "Read next" lists.

## Chapter guide

### Part I — Foundations

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 1 | The maths you actually use | Vectors, SVD and PCA, probability, MLE and MAP (squared error and ridge derived), gradients, matrix calculus and curvature (2/λ_max, condition number), entropy and KL (forward vs reverse), perplexity and bits per byte, Monte Carlo, importance sampling and the score-function trick, Jensen and the ELBO | 6–8 |
| 2 | Statistics and experimentation | Confidence intervals and the delta method, tests, power, A/B test design (triggering, A/A tests, LLM features), CUPED, multiple testing, sequential tests, causal inference (potential outcomes, DAGs, AIPW), statistics for model evaluation (pairing, Bradley–Terry, seeds), deciding under uncertainty | 6–8 |
| 3 | ML fundamentals | Framing, bias–variance, losses, regularisation, optimisation basics, splits, leakage, calibration | 5–7 |
| 4 | Evaluation and metrics | Classification, ranking and regression metrics, calibration, slicing, confidence intervals on metrics | 4–6 |

### Part II — Deep learning and training dynamics

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 5 | Neural network fundamentals | MLPs, backpropagation by hand, activations, initialisation, normalisation, debugging | 4–6 |
| 6 | PyTorch for practitioners | Tensors and autograd, modules, the training loop, mixed precision, checkpointing, reproducibility | 4–6 |
| 7 | Optimisation and training dynamics | AdamW, schedules (cosine, WSD), batch size and gradient noise, Muon and other optimisers, µP, bf16/fp8, loss spikes, z-loss, QK-norm, reading training curves | 6–8 |
| 8 | Embeddings and sequence models | word2vec, RNNs, LSTMs, seq2seq and the first attention; the link to modern recurrent models | 3–4 |

### Part III — Transformers and architectures

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 9 | Text and tokenisation | BPE from scratch, tokenizer trade-offs, multilingual cost, tokenizer research (vocabulary scaling, byte-level models) | 4–5 |
| 10 | The transformer in depth | Attention, GQA, RoPE, normalisation, SwiGLU, the KV cache, a mini-GPT; long-context extension (PI, NTK, YaRN), attention sinks | 7–9 |
| 11 | Architectures beyond the vanilla transformer | Mixture of experts and load balancing, MLA, sparse and sliding attention, linear attention and SSMs, hybrids, multi-token prediction, diffusion LMs | 5–7 |
| 12 | Model families and scaling | Encoders, decoders and encoder–decoders; sentence transformers; MoE and scaling in overview; choosing a model | 3–4 |
| 13 | Scaling laws | Kaplan vs Chinchilla, deriving the compute-optimal allocation, IsoFLOP fits, over-training for inference, repeated data, emergence, test-time compute | 5–7 |
| 14 | Multimodal models | ViT, CLIP and SigLIP, VLM architectures and training, resolution and token budgets, speech, image generation, video, multimodal evaluation | 4–6 |

### Part IV — Pretraining

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 15 | How LLMs are trained | The pipeline end to end: pretraining, SFT, RLHF, DPO, RLVR and distillation, and what each stage explains | 3–4 |
| 16 | Pretraining data | Web pipelines, heuristic and model-based filtering, MinHash deduplication, decontamination, mixtures, annealing and mid-training, synthetic data, data ablations | 4–6 |
| 17 | Pretraining at scale | FLOPs, time and MFU; memory anatomy; ring all-reduce, ZeRO and FSDP; tensor, pipeline, context and expert parallelism; checkpointing and failures | 6–8 |

### Part V — Post-training

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 18 | Post-training I: SFT and preferences | SFT data, masking and packing, reward models, policy-gradient foundations (REINFORCE, baselines, importance sampling), PPO with GAE, KL estimators, over-optimisation, DPO derivation and variants, AI feedback | 7–9 |
| 19 | Post-training II: RL and reasoning | RL with verifiable rewards, GRPO and its fixes (Dr. GRPO, DAPO), verifiers and reward hacking, curricula, does RL add reasoning, PRMs, test-time strategies, distilling reasoning, RL systems | 7–9 |
| 20 | Fine-tuning and parameter-efficient methods | When to fine-tune, LoRA and QLoRA in depth, adapters, instruction data, forgetting, continued pretraining, model merging, RL fine-tuning | 4–6 |

### Part VI — Inference and prompting

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 21 | Inference, decoding and serving | Decoding strategies, the KV cache, prefill and decode, serving engines, speculative decoding, quantisation, cost arithmetic, roofline, disaggregation, serving MoE and reasoning models | 6–8 |
| 22 | Prompting and context engineering | Prompt anatomy, few-shot, reasoning, structured output, long context, caching, prompt injection, in-context learning research | 4–6 |

### Part VII — Evaluation science

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 23 | Evaluating models | Error bars, clustering and paired tests, sample size, sources of variance, bits per byte, benchmark design and IRT, contamination, arenas, judge biases, agent horizons | 5–7 |
| 24 | LLM evaluation | Evaluating LLM applications: eval sets, task metrics, LLM judges and their calibration, regression testing | 4–6 |
| 25 | RAG and agent evaluation | Retrieval and faithfulness metrics, end-state and trajectory evaluation, pass^k reliability | 3–5 |

### Part VIII — Interpretability, alignment and safety

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 26 | Interpretability and analysis | Residual stream view, probing, logit lens, activation patching, induction circuits, superposition and sparse autoencoders, circuit tracing, steering | 5–7 |
| 27 | Alignment and safety research | Reward hacking, sycophancy, jailbreaks and defences, automated red-teaming, scalable oversight and weak-to-strong, deception evaluations, control and CoT monitoring, dangerous-capability evaluations | 4–6 |
| 28 | Safety and responsible AI | Responsible AI in products: moderation, privacy, fairness, injection defences, regulation | 3–5 |

### Part IX — Retrieval and agents

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 29 | Embeddings and representation learning | Contrastive learning, hard negatives, fine-tuning embedders, Matryoshka, quantisation, LLM-based embedders | 4–6 |
| 30 | Information retrieval | BM25, dense retrieval, ANN indexes, filtering, hybrid search, re-ranking, query rewriting, retrieval evaluation, learned sparse and reasoning-intensive retrieval | 5–7 |
| 31 | Retrieval-augmented generation | Chunking, retrieval pipelines, citations, failure modes and fixes | 3–4 |
| 32 | Frameworks and tooling | Application frameworks (LangChain, LangGraph, LlamaIndex, DSPy) and the research stack (Megatron, torchtitan, TRL, verl, vLLM, eval harnesses) | 2–3 |
| 33 | Agents and tool use | Workflows vs agents, function calling, planning, memory, MCP, guardrails, reliability; training agents with RL | 4–6 |

### Part X — Research craft and interviews

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 34 | Research craft | Choosing problems, reading and reproducing papers, experiment design and ablations, seeds and statistics, compute budgeting, writing research memos | 4–5 |
| 35 | The ML coding interview | pandas and SQL patterns, classic algorithms, and from-scratch components for research rounds (attention, losses, samplers, routers) | 6–8 |
| 36 | Research interviews | The research-design round with worked answers, paper discussions, research talks, take-home tasks | 4–5 |
| 37 | ML depth question bank | 260 questions with short answers, including 100 on LLM research and RL; follow-up chains and graded answers | 8–12 |
| 38 | Behavioural interviews | STAR, a story bank, senior signals, research stories | 3–4 |
| 39 | Thinking like a senior | Trade-offs, scoping, uncertainty, influence, research judgement | 3–4 |

### Part XI — Applied ML systems (optional track)

| Ch | Title | What it covers | Hours |
|---|---|---|---|
| 40 | Classic models | Linear and logistic regression, trees, forests, boosting, kNN, SVMs, naive Bayes, k-means, PCA, from scratch | 5–7 |
| 41 | NLP tasks with modern tools | Classification with LLMs and distillation, NER, sentiment, topics, summarisation, translation, task metrics | 4–6 |
| 42 | Infrastructure primer | Servers, GPUs, APIs, Docker, Kubernetes, queues, batch vs online inference, capacity planning | 3–4 |
| 43 | Recommender systems | Collaborative filtering, two-tower retrieval, ranking, bandits, position bias, off-policy evaluation, generative recommenders | 6–8 |
| 44 | Search and learning to rank | Query understanding, BM25, LTR (RankNet to LambdaMART), fusion, LLM rerankers | 5–7 |
| 45 | Marketplace and logistics ML | Forecasting, ETA, matching, pricing | 3–5 |
| 46 | ML system design | The eight-step method, back-of-the-envelope maths, LLM systems, research system design | 6–8 |
| 47 | MLOps and LLMOps | Pipelines, registries, release ladders, monitoring, LLMOps, research-scale operations | 3–4 |
| 48 | Data for ML | SQL patterns, point-in-time joins, Spark, data contracts, labelling | 3–4 |

### Appendices

- **A.** Glossary
- **B.** One-page cheat sheets, including LLM research
- **C.** Notation and formulas
- **D.** Paper map: the papers worth knowing, from foundations to the end of 2025
- **E.** Ten worked system designs, the model answers for Chapter 46
- **F.** A 48-question diagnostic with an answer key: take it first, and again after four weeks

## Study plans

**Start with the diagnostic.** Appendix F has 48 questions, five to seven per area, with an answer key and a rule for turning the scores into a plan. Use the plans below as the default order, then spend extra time on the areas where you scored 0–4 and skip ahead where you scored 8–10.

**If the diagnostic shows a gap in Part I** (a score of 0–4 on F.1), fix it first, because every later chapter assumes Chapters 1–4. Work through them in full, including the derivations, before starting either track. If the gap is in the underlying mathematics rather than its use in ML (you cannot multiply matrices by hand, or conditional probability is shaky), spend a week first on *Mathematics for Machine Learning* (chapters 2, 5 and 6: linear algebra, vector calculus, probability) and 3Blue1Brown's linear-algebra series, then return. Go as deep as each chapter's "Done when" list and no deeper: interviews ask you to derive the specific results in these chapters, not to prove general theorems.

### How long each chapter takes

The **Hours** column in the chapter guide is a budget for one careful pass: reading with the worked examples recomputed, running the chapter's script and changing something, answering the interview questions aloud and doing the practice items. It does not include the research exercise, which takes another 2–6 hours. The estimates are by density rather than length: derivation-heavy chapters (1, 2, 7, 10, 18, 19) take longest per page, while surveys and maps (12, 15, 32) go quickly. To **skim** a chapter means its Key concepts, the opening paragraph of each section, the Pitfalls and the Interview questions: 1–2 hours. Halve the budget for a chapter you already know well and only need to check against its "Done when" list.

### Research track: twelve weeks to LLM research-scientist readiness

Assumes you know ML and some deep learning. Expect about 15–20 hours a week: the main chapters below plus the interview thread that starts in week 5. Do one research exercise a week from that week's chapters (2–6 hours on top); they are the main practice. With 8–10 hours a week, keep the same order and stretch it over 20–24 weeks.

| Week | Read in full | Skim | Focus | Hours |
|---|---|---|---|---|
| 1 | 1, 2 | | The maths and statistics you will be asked to derive; evaluation statistics (2.11) | 12–16 |
| 2 | 3, 4, 5 | | Bias–variance, leakage (contamination is leakage), calibration, metrics with confidence intervals, backpropagation | 13–19 |
| 3 | 6, 7 | 8 | PyTorch mechanics and training dynamics | 11–16 |
| 4 | 9, 10 | | Tokenisation and the transformer in depth | 11–14 |
| 5 | 11, 13 | 12 | Architectures and scaling laws | 11–16 |
| 6 | 16, 17 | 14, 15 | Pretraining data and systems (read 14 in full for multimodal roles) | 12–18 |
| 7 | 18, 19 | | Post-training and reasoning (the densest week) | 14–18 |
| 8 | 20, 21 | | Fine-tuning and inference | 10–14 |
| 9 | 22, 23 | | Prompting and context engineering; evaluation science | 9–13 |
| 10 | 24, 25, 26 | | Application and agent evaluation; interpretability | 12–18 |
| 11 | 27, 30 | 28 | Alignment and safety; retrieval | 10–15 |
| 12 | 29, 33 | 31, 32 | Embeddings and agents; final review with Appendices B and D | 10–16 |

**The interview thread, in parallel from week 5** (4–5 hours a week, 30–40 hours in all):

| Weeks | Chapters | What to do |
|---|---|---|
| 5–6 | 34 | Research craft; choose a small research project you can finish in four weeks |
| 5–12 | 35, 37 | One from-scratch component a week (35.9), timed; the question-bank section for each part you have finished, aloud |
| 7–8 | 36 | The research-design round and paper discussion; start preparing a 20-minute talk on your project |
| 9–10 | 38 | Write five stories in STAR form, at least two of them research stories |
| 11–12 | 39 | Senior judgement; two full mock loops, one with the talk |

### Applied track: eight weeks for applied ML and LLM engineering roles

| Week | Chapters | Focus | Hours |
|---|---|---|---|
| 1 | 46, 43 | The design framework and recommender systems; start weekly timed design practice | 12–16 |
| 2 | 4, 44 | Metrics and search ranking | 9–13 |
| 3 | 30, 41, 22 | Retrieval, LLM classification, prompting | 13–19 |
| 4 | 31, 24, 25 | RAG and its evaluation | 10–15 |
| 5 | 3, 40, 10 | ML fundamentals and the transformer | 17–23 |
| 6 | 29, 20, 33 | Embeddings, fine-tuning, agents | 12–18 |
| 7 | 2, 45, 38 | Experimentation, marketplace ML, behavioural stories | 12–17 |
| 8 | 35, 37, Appendix B | The mock loop | 14–20 |

**Two-week refresh** (you have done the work before and the interview is close). Week 1: take the diagnostic, then each day one red or amber area: its chapters' pitfalls, interview questions and "Done when" box, plus the matching section of the question bank aloud. Week 2: one timed design (Appendix E) and one from-scratch coding component (Chapter 35.9) each day, the follow-up chains in Chapter 37, and your story bank (Chapter 38) rehearsed aloud.

**Seventy-two hours.** Day 1: the cheat sheets (Appendix B) and the question bank's senior questions in your target area, aloud and timed. Day 2: two timed system designs and two coding components; the follow-up chains. Day 3: five stories in STAR form, your one-minute introduction (Chapter 39.7), questions to ask your interviewers (Chapter 36.7), and sleep.

## Interview loops and where to prepare

Loops differ by company and change often, so treat this as a map of typical rounds rather than a description of any one employer. Find the row closest to your target role, check the rounds against what your recruiter tells you, and weight your plan towards the chapters in that row. Chapter 36.1 describes the research loop in more detail.

| Loop | Typical rounds | Chapters and appendices | What "strong" looks like |
|---|---|---|---|
| Frontier-lab research scientist | Research design; paper discussion; research talk; coding of model components; ML depth; behavioural, including motivation for the lab's mission and safety | 34, 36, 35.9, 37, 13, 18, 19, 23, 26, 27, 38; B, D | Proposes a falsifiable experiment with baselines, a compute budget and a kill criterion; derives results such as the policy gradient or DPO on request; puts error bars on every comparison |
| Frontier-lab research engineer (training or inference systems) | Coding, including from-scratch components and debugging a broken training loop; systems design for training or serving; performance reasoning | 6, 7, 10, 17, 21, 35, 46 (LLM and research system design); E | Back-of-the-envelope memory, FLOPs and latency numbers within a factor of two; finds bugs systematically; knows the parallelism, KV-cache and MFU trade-offs |
| Big-tech applied scientist | ML breadth and depth; ML system design; coding (algorithms and ML); a research presentation or paper discussion; behavioural | 2, 3, 4, 40, 43, 44, 46, 35, 37, 38; E | Frames the business problem, chooses metrics and an experiment, justifies a model ladder, and goes deep on one model when pushed |
| Big-tech ML engineer | Coding (data structures and algorithms); ML system design; ML fundamentals; behavioural | 35, 46, 42, 47, 48, 3, 4; E | Correct, tested code at speed; a design that covers serving, monitoring and data contracts; sensible capacity numbers |
| Applied AI or LLM engineer | LLM application design (RAG, agents, evaluation); prompting and context engineering; coding; often a take-home | 20, 21, 22, 24, 25, 29, 30, 31, 33, 46 (LLM systems); E.4, E.10 | Starts from an evaluation set and error analysis; compares prompting, retrieval and fine-tuning with cost and latency numbers; designs guardrails |
| Product data scientist (experimentation) | Statistics and an A/B-test case; SQL; product sense and metrics | 2, 4, 48, 35.4, 45 | Sample size, SRM, interference and a ship decision under uncertainty, each with a number |

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
