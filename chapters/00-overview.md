# Overview

A self-study guide for data scientists and ML engineers preparing for senior applied ML and LLM roles: the theory, the code, the system design and the interview, in one place. Examples come from food delivery and marketplaces, but every chapter applies to any product company.

**All chapters:** see the [linked contents in the README](../README.md#contents).

## How the handbook works

The guide has 35 chapters in 10 parts (including Chapter 13a, an infrastructure primer for readers new to production systems) and five appendices. Read it in order, or jump to the part you need; a suggested study plan is below.

**Every chapter follows the same template:**

1. **Why it matters.** Where this shows up on the job and in interviews, in three or four lines.
2. **Core concepts.** Each concept goes intuition first, then the maths, then working code (Python, NumPy, PyTorch or Hugging Face).
3. **Diagrams.** Architectures, data flows and pipelines drawn, not described.
4. **Pitfalls.** The mistakes practitioners make and interviewers probe.
5. **Interview questions.** 10–20 per chapter with model answers, from basic to senior-level.
6. **Practice.** Coding exercises to do yourself, with where to practise (Deep-ML, LeetCode pandas plans, Kaggle datasets, your own repo) and what a good solution looks like.
7. **Key concepts and Read next.** Every chapter opens with a list of its key concepts, models and architectures, each explained in a line with a link to go deeper, and ends with papers, docs and books for further reading.
8. **Done when.** A checklist of what you should be able to explain or build without notes.

**How to use it:** read a section, close it, explain it aloud, then do the practice. If you use an AI assistant, paste your code or explanation and ask it to review your work against the chapter. Code examples are written for understanding first; they run, but they are not production-hardened.

**A caution on fast-moving tools:** library APIs (LangChain, LangGraph, PEFT, vLLM) change every few months. Concepts in the book are stable; API calls are dated, and you should check them against the current docs before relying on them.

## Contents, Parts I–III

Parts I–III are the base: the maths, statistics, classic models and deep learning that every later chapter assumes.

### Part I — Foundations

| Ch | Title | What it covers |
|---|---|---|
| 1 | The maths you actually use | Vectors, dot products and cosine similarity; matrix multiplication; eigenvectors, SVD and low-rank approximation (the basis of LoRA and matrix factorisation); probability, Bayes, common distributions; MLE and MAP; gradients and the chain rule; entropy, cross-entropy and KL divergence |
| 2 | Statistics and experimentation | Sampling, confidence intervals, the bootstrap; hypothesis tests and p-values; power and sample size; A/B test design, guardrail metrics, novelty effects; CUPED; multiple testing; sequential testing; interleaving for ranking; switchback tests in marketplaces; causal inference basics (DiD, IPW, uplift) |
| 3 | ML fundamentals | Framing a problem as ML; bias-variance; loss functions; regularisation (L1, L2, early stopping); optimisation (GD, SGD, momentum, Adam); train/validation/test, cross-validation, time-based splits; leakage; feature engineering; class imbalance; calibration |
| 4 | Classic models | Linear and logistic regression; decision trees; random forests; gradient boosting (XGBoost, LightGBM, CatBoost); kNN; SVMs; naive Bayes; k-means; PCA. Each with intuition, maths, a from-scratch implementation and when to use it |
| 5 | Evaluation and metrics | Classification (precision, recall, F1, ROC-AUC, PR-AUC, log loss, calibration curves); regression (MAE, RMSE, MAPE); ranking (precision@k, recall@k, MAP, MRR, NDCG); offline vs online metrics; slicing; confidence intervals on metrics |

### Part II — Deep learning

| Ch | Title | What it covers |
|---|---|---|
| 6 | Neural network fundamentals | Perceptron to MLP; backpropagation by hand; activations; initialisation; batch and layer normalisation; dropout; optimisers and learning-rate schedules; mixed precision; debugging a model that won't learn |
| 7 | PyTorch for practitioners | Tensors and autograd; nn.Module; the training loop; Dataset and DataLoader; GPU basics; checkpointing; reproducibility |
| 8 | Embeddings and sequence models | word2vec, GloVe, fastText; RNN, LSTM, GRU; seq2seq; the first attention mechanism (Bahdanau) and why it led to transformers |

### Part III — NLP and transformers

| Ch | Title | What it covers |
|---|---|---|
| 9 | Text and tokenisation | Normalisation; BPE, WordPiece, unigram and SentencePiece; vocabulary trade-offs; multilingual and code-switching issues; token counting and cost |
| 10 | The transformer in depth | Scaled dot-product and multi-head attention; positional encodings (sinusoidal, learned, RoPE, ALiBi); residuals and pre- vs post-LayerNorm; feed-forward blocks and SwiGLU; encoder, decoder and encoder-decoder; complexity and the KV cache; attention variants (MQA, GQA, FlashAttention, sliding window). Code: attention from scratch and a mini-GPT |
| 11 | Model families and scaling | BERT and masked LM; GPT and causal LM; T5; sentence transformers; mixture of experts; vision-language models (CLIP and successors); scaling laws (Kaplan, Chinchilla); small vs large models in practice |
| 12 | NLP tasks with modern tools | Text classification (an LLM merchant classifier as a worked case); NER and extraction; sentiment; topic modelling (LDA to BERTopic); summarisation; translation; task metrics (BLEU, ROUGE, BERTScore) and their limits |

## Contents, Parts IV–VII

Parts IV–VII are the core of the book and the largest share of it: how LLMs work, how to adapt them, how to ground them in data, and how to measure them.

### Part IV — Large language models

| Ch | Title | What it covers |
|---|---|---|
| 13 | How LLMs are trained | Pretraining data and objective; supervised fine-tuning; RLHF (reward models, PPO); DPO and its variants; RLAIF and constitutional approaches; reasoning models and RL with verifiable rewards; distillation into small models |
| 14 | Inference, decoding and serving | Greedy, beam, temperature, top-k, top-p, min-p; repetition penalties; constrained and structured decoding; speculative decoding; KV cache and continuous batching; quantisation (int8, int4, GPTQ, AWQ); serving engines (vLLM, TGI); latency, throughput and cost arithmetic; batch APIs (cutting an offline LLM job's cost as a worked case) |
| 15 | Prompting and context engineering | Zero- and few-shot; chain of thought; structured output (JSON schema, function calling); prompt templates and versioning; long context and its failure modes; prompt caching; prompt injection and defences |
| 16 | Fine-tuning and parameter-efficient methods | When to prompt, retrieve or fine-tune; full fine-tuning; LoRA in depth (the low-rank update, rank and alpha, target modules, merging); QLoRA; adapters; prefix-tuning and prompt tuning; instruction data; synthetic data; catastrophic forgetting; evaluating a fine-tune. Code: LoRA from scratch, then with Hugging Face PEFT |
| 17 | Embeddings and representation learning | Sentence embeddings; contrastive learning and InfoNCE; hard negatives; fine-tuning an embedding model; Matryoshka embeddings; multilingual embeddings; benchmarks (MTEB) and why your own evaluation matters more |

**Chapter 13a · Infrastructure primer.** Servers, CPUs and GPUs, HTTP APIs, Docker, Kubernetes, storage, queues and streams, batch vs online inference, capacity planning and observability, explained from zero for readers who have never deployed a model. Read it before Chapters 14 and 29.

### Part V — Retrieval and RAG

| Ch | Title | What it covers |
|---|---|---|
| 18 | Information retrieval | Inverted index; TF-IDF and BM25; dense retrieval (DPR); approximate nearest neighbour (HNSW, IVF, product quantisation); vector databases; hybrid search and reciprocal rank fusion; re-ranking with cross-encoders; late interaction (ColBERT); query understanding and rewriting; retrieval metrics |
| 19 | Retrieval-augmented generation | The RAG pipeline end to end; chunking strategies; metadata and filtering; query transformation (multi-query, HyDE); context assembly and ordering; citations and grounding; GraphRAG; agentic RAG; common failure modes and fixes |
| 20 | Frameworks and tooling | LangChain (core abstractions, LCEL, retrievers, tools); LangGraph (state graphs for agents); LlamaIndex; DSPy (programming prompts as modules); the Hugging Face stack; when a framework helps and when plain Python is better |

### Part VI — Agents

| Ch | Title | What it covers |
|---|---|---|
| 21 | Agents and tool use | Function calling; ReAct; planning and reflection; memory; multi-agent patterns; the Model Context Protocol; guardrails; reliability, cost and latency budgets; evaluating agents (task success, trajectories, tool-call accuracy) |

### Part VII — Evaluating LLM systems

| Ch | Title | What it covers |
|---|---|---|
| 22 | LLM evaluation | Benchmarks vs application evals; error analysis as the starting point; building an eval set; reference-based metrics (exact match, token F1, BLEU, ROUGE, BERTScore); LLM-as-judge (pointwise, pairwise, rubric-based); judge biases (position, verbosity, self-preference) and how to calibrate judges against human labels (agreement, Cohen's kappa); sample sizes and confidence intervals; online evaluation and A/B tests for LLM features; tooling |
| 23 | RAG and agent evaluation | Retrieval metrics vs generation metrics; context precision and recall; faithfulness and answer relevance; RAGAS-style evaluation and its pitfalls; evaluating multi-step agents |
| 24 | Safety and responsible AI | Hallucination; toxicity and bias; PII handling; prompt injection and jailbreaks; red-teaming; content moderation (a review moderation pipeline as a worked case); the EU AI Act at a practitioner level |

## Contents, Parts VIII–X

Parts VIII–X turn knowledge into interview performance: the recommender and search depth that consumer-product roles need, system design, production practice, and the interview itself.

### Part VIII — Recommender and search systems

| Ch | Title | What it covers |
|---|---|---|
| 25 | Recommender systems | Framing (what to optimise, implicit vs explicit feedback); collaborative filtering and matrix factorisation; the multi-stage architecture (candidate generation, ranking, re-ranking); two-tower retrieval; ranking models (GBDT, DLRM, DCN, sequential models such as SASRec); cold start; exploration and bandits; position bias and inverse propensity scoring; offline replay and its limits; LLMs in recommendation (LLM judges, semantic IDs, generative retrieval) |
| 26 | Search ranking and learning to rank | Pointwise, pairwise and listwise losses; LambdaMART; query, document and query-document features; marketplace search (availability, distance, delivery time); query autocomplete and spelling; evaluating search |
| 27 | Marketplace and logistics ML | Demand forecasting (from baselines to gradient boosting and deep models); ETA prediction; supply-demand balancing and network rebalancing (optimisation basics: linear programming, assignment); pricing and incentives at a conceptual level; experiments in two-sided marketplaces |

### Part IX — ML systems and production

| Ch | Title | What it covers |
|---|---|---|
| 28 | ML system design | The answer framework (clarify, frame, data, features, model, evaluation, serving, monitoring); back-of-the-envelope maths; then 10 worked designs: restaurant feed, dish and restaurant search, LLM taxonomy classifier, RAG support assistant, review moderation and summarisation, LLM judge for relevance, ad click prediction, demand forecasting for quick commerce, fraud detection, an agentic customer-service flow |
| 29 | MLOps and LLMOps | Feature stores; training pipelines; model registry; CI/CD for models; shadow, canary and A/B deployment; monitoring and drift; LLM observability (tracing chained calls, tokens, cost); prompt and model versioning |
| 30 | Data for ML | SQL patterns for ML (windows, sessionisation, point-in-time joins); pandas at scale; Spark basics; data quality and contracts; labelling strategies and inter-annotator agreement; synthetic data |

### Part X — Interview craft

| Ch | Title | What it covers |
|---|---|---|
| 31 | The ML coding interview | The common task formats; pandas and SQL patterns; a list of algorithms to implement from scratch; object-oriented code for ML; testing; a problem bank with where to practise each |
| 32 | ML depth question bank | 150+ questions with model answers, grouped by chapter, marked by level |
| 33 | Behavioural interviews | The STAR structure; your five core stories written out; what senior signals interviewers listen for; questions to ask them |
| 34 | Thinking like a senior | Trade-off reasoning; product sense for ML; scoping and saying no; communicating uncertainty; leading a project without authority |

### Appendices

- **A.** Glossary
- **B.** One-page cheat sheets per part, for the night before
- **C.** Notation and common formulas
- **D.** Paper map: the 40 papers worth knowing, one paragraph each
- **E.** Ten worked system designs, the model answers for Chapter 28

## A suggested study plan

If you are preparing for interviews, this eight-week order front-loads what interview loops test most; the builder-depth chapters follow. Reorder it for your target role.

| Week | Chapters | Why then |
|---|---|---|
| 1 | 28 ML system design, the framework part; 25 Recommender systems | Start the weekly timed design practice with a restaurant feed (Chapter 28) |
| 2 | 5 Evaluation and metrics; 26 Search ranking | Ranking metrics and the search design exercise |
| 3 | 18 Information retrieval; 12 NLP tasks; 15 Prompting | Two-tower retrieval and the LLM classification design |
| 4 | 19 RAG; 22 LLM evaluation; 23 RAG and agent evaluation | The RAG assistant design and the first mock |
| 5 | 3 ML fundamentals; 4 Classic models; 10 The transformer | The ML depth refresh week |
| 6 | 17 Embeddings; 16 Fine-tuning and LoRA; 21 Agents | Semantic IDs and the LLM-judge design |
| 7 | 2 Statistics and experimentation; 27 Marketplace ML; 33 Behavioural | Experimentation, forecasting and your stories |
| 8 | 31 Coding interview; 32 Question bank; Appendix B cheat sheets | The mock loop |
| 9 onwards | 1, 6, 7, 8, 9, 11, 13, 13a, 14, 20, 24, 29, 30, 34; Appendices A, C, D | Builder depth and infrastructure (read 13a early if production systems are new to you) |

At about three chapters a week this is a lot of reading. If a week gets heavy, read the "Core concepts" and "Interview questions" sections first and leave "Read next" for later.

## Core library

Two books are worth owning; everything else here is free or optional. Each chapter's "Read next" points to specific chapters or sections of these.

### Books

| Book | Used in chapters | Cost |
|---|---|---|
| *Machine Learning System Design Interview*, Aminian and Xu | 25, 26, 28 | Worth buying first |
| *AI Engineering*, Chip Huyen (2024) | 14, 15, 16, 19, 21, 22, 29 | Worth buying first |
| *Designing Machine Learning Systems*, Chip Huyen (2022) | 3, 28, 29, 30 | Optional |
| *An Introduction to Statistical Learning*, James, Witten, Hastie, Tibshirani | 3, 4, 5 | Free PDF from the authors |
| [*Dive into Deep Learning*](https://d2l.ai/), Zhang et al. | 6, 7, 8, 10 | Free online |
| [*Speech and Language Processing* (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/), Jurafsky and Martin | 8, 9, 10, 12, 18 | Free online |
| *Hands-On Large Language Models*, Alammar and Grootendorst | 11, 12, 16, 17, 19 | Optional |
| *Build a Large Language Model (From* *Scratch)*, Raschka | 10, 13, 16 | Optional |
| [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/), Manning, Raghavan, Schütze | 18, 26 | Free online |
| *Trustworthy Online Controlled Experiments*, Kohavi, Tang, Xu | 2 | Optional |

### Blogs and courses

- [Andrej Karpathy, Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — chapters 6, 9, 10
- [Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — chapter 10
- [Lilian Weng's blog](https://lilianweng.github.io/) — chapters 13, 16, 21
- [Eugene Yan's writing](https://eugeneyan.com/) — chapters 22, 25, 26, 28
- [Hamel Husain's blog](https://hamel.dev/) — chapters 22, 23

### Papers worth reading in full

| Paper | Chapter |
|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) | 10 |
| [BERT](https://arxiv.org/abs/1810.04805) (2018) | 11 |
| [Sentence-BERT](https://arxiv.org/abs/1908.10084) (2019) | 17 |
| [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (2020) | 18 |
| [ColBERT](https://arxiv.org/abs/2004.12832) (2020) | 18 |
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (2020) | 19 |
| [Prefix-Tuning](https://arxiv.org/abs/2101.00190) (2021) | 16 |
| [LoRA](https://arxiv.org/abs/2106.09685) (2021) | 16 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) | 13 |
| [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) (2022) | 11 |
| [ReAct](https://arxiv.org/abs/2210.03629) (2022) | 21 |
| [Recommender Systems with Generative Retrieval (semantic IDs)](https://arxiv.org/abs/2305.05065) (2023) | 25 |
| [QLoRA](https://arxiv.org/abs/2305.14314) (2023) | 16 |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (2023) | 13 |
| [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) (2023) | 22 |
| Deep Neural Networks for YouTube Recommendations, Covington et al. (RecSys 2016) | 25 |

The arXiv IDs are the papers' standard identifiers.

---

[Contents](../README.md#contents) · [01 · The maths you actually use →](01-the-maths-you-actually-use.md)
