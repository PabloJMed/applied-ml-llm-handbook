# F · Diagnostic self-assessment

Forty questions that tell you where to spend your preparation time. Take it before you start a study plan and again four weeks later. It takes about 90 minutes.

**How to take it.** Closed book, no assistant, a timer at two minutes per question. Answer aloud or in writing, as you would in an interview: the reasoning counts, not only the final number. Then score yourself against the answer key at the end of this appendix:

| Score | Meaning |
|---|---|
| 2 | You gave every point in the key, or an equally good answer, within two minutes |
| 1 | You had the main idea but missed a key point, made an arithmetic slip or needed far more than two minutes |
| 0 | Wrong, blank, or the right words without being able to explain them |

Be strict. A 1 you scored as a 2 is a gap you will meet in the interview instead.

**Turning scores into a plan.** Each area has five questions, so a score out of 10:

| Area score | What to do with that area's chapters |
|---|---|
| 0–4 | Study in full: read, run the script, do the practice and the research exercise |
| 5–7 | Read the pitfalls and interview questions, redo the practice for the questions you missed, then drill the matching section of the question bank (Chapter 37) |
| 8–10 | Check the "Done when" box, then move on; revisit only in the final week |

Order the areas that scored 0–4 by how much your target role weighs them (the research and applied tracks in the Overview show the weights) and give each one a week. Most people find one or two areas they thought were strong scoring 5–7: those are the cheapest points to win.

## F.1 Foundations (Chapters 1–4)

**1.** A fraud detector has 95% recall and a 1% false-positive rate. Fraud is 0.1% of transactions. What fraction of flagged transactions are fraud?

**2.** Training with cross-entropy minimises a KL divergence. Which one, in which direction, and what does that direction do to the model's behaviour? Where does the other direction appear in LLM training?

**3.** A colleague checks an A/B test every day and stops when p < 0.05. What goes wrong, roughly how badly, and what should they do instead?

**4.** On the same 500 test items, model A scores 82% and model B 84%. Is B better? How would you decide?

**5.** A model has 0.97 validation AUC and performs badly in production. List the causes you would check, in order.

## F.2 Deep learning and training dynamics (Chapters 5–8)

**6.** A 10-class classifier's loss at step 0 is 7.5. What should it be, and what does 7.5 suggest?

**7.** Write the AdamW update. Why is its weight decay "decoupled", and why does that matter?

**8.** Why does Adam need learning-rate warm-up? What does a warmup–stable–decay (WSD) schedule give you that cosine does not?

**9.** Why is bf16 preferred to fp16 for training, and why keep fp32 master weights anyway?

**10.** You double the width of a model under the standard parametrisation and keep the learning rate. What happens, and how does µP change it?

## F.3 Transformers, architectures and scaling (Chapters 9–14)

**11.** Estimate the KV cache for one 8,192-token sequence: 32 layers, 8 KV heads, head dimension 128, bf16. How do GQA and MLA change it?

**12.** Why is the attention score divided by √d_k?

**13.** How does RoPE encode position? Why does a model trained at 8k tokens degrade at 32k, and what are two fixes?

**14.** You have 1e23 training FLOPs. Using Chinchilla's rule of thumb, what model size and token count are compute-optimal? Why do most open models train far smaller models on far more tokens?

**15.** A mixture-of-experts layer has 64 experts with top-2 routing. What scales with the number of experts and what with k? Why does it need load balancing, and how is it done?

## F.4 Pretraining (Chapters 15–17)

**16.** How much GPU memory does training a 7B-parameter model with Adam in mixed precision need, excluding activations? How does ZeRO-3 on 8 GPUs change that?

**17.** Estimate the time to train a 7B model on 2T tokens on 256 H100s (989 TFLOP/s dense bf16 peak) at 40% MFU.

**18.** How does MinHash deduplication find near-duplicate documents, and why does deduplication improve models?

**19.** Data, tensor and pipeline parallelism: what does each split, what does each communicate, and where in a cluster does each run?

**20.** A new model scores far above its predecessor on a public benchmark. How would you check for contamination?

## F.5 Post-training and RL (Chapters 18–20)

**21.** Derive the policy gradient for J(θ) = E_(y∼π_θ)[R(y)]. Why does subtracting a baseline leave it unbiased?

**22.** Derive the DPO loss from the KL-regularised RL objective in a few lines.

**23.** A GRPO group of four responses gets rewards [1, 0, 0, 0]. What are the advantages? What happens when all four rewards are equal, and why does it matter for the prompt set?

**24.** What is reward over-optimisation? How would you detect it and limit it?

**25.** How many trainable parameters does a rank-16 LoRA add to a 4096 × 4096 weight matrix? Why does such a small update work, and what does QLoRA add?

## F.6 Inference, prompting and evaluation (Chapters 21–25)

**26.** Why is the prefill phase compute-bound and the decode phase memory-bandwidth-bound? What does that imply for batching?

**27.** How does speculative decoding produce exactly the target model's distribution? When does it stop helping?

**28.** When does majority voting over sampled answers (self-consistency) make accuracy worse?

**29.** You want an LLM judge to score answers in your application. How do you show the judge can be trusted?

**30.** An agent succeeds at each step with probability 0.95. What is its success rate on a 20-step task, and what does that imply for how you evaluate and design agents?

## F.7 Interpretability, alignment, retrieval and agents (Chapters 26–33)

**31.** A linear probe recovers "is this sentence negated" from layer 12 with 95% accuracy. What does that show and not show? How would activation patching add to it?

**32.** What is an induction head, and what two components form the circuit?

**33.** When does BM25 beat dense retrieval and vice versa? How would you combine them?

**34.** A RAG system answers a question wrongly although the right document is in the index. In what order do you debug?

**35.** An agent reads a user's emails and can send messages. How would you defend against indirect prompt injection?

## F.8 Applied systems and senior judgement (Chapters 34–48)

**36.** A new ranker improves offline NDCG by 3%, but the A/B test is flat. Give the likely reasons, in order.

**37.** Why do large recommender systems use separate retrieval and ranking stages? What model is typical for each?

**38.** What is a point-in-time join, and what goes wrong in training data without one?

**39.** "Tell me about a time you were wrong." What separates a senior answer from a mid-level one?

**40.** A product manager asks for a classifier with 99% accuracy by next month. How do you respond?

## F.9 Answer key

The points below are what a full-marks answer contains. The chapter in brackets has the full treatment.

**1.** Bayes: true positives 0.001 × 0.95 = 0.00095; false positives 0.999 × 0.01 = 0.00999; precision 0.00095 / 0.01094 ≈ **8.7%**. At low prevalence, precision is dominated by the false-positive rate; report precision at the real base rate, not on a balanced test set (Ch 4).

**2.** Cross-entropy on data = H(p_data) + KL(p_data ‖ p_model); the entropy is fixed, so training minimises **forward KL**. Forward KL is mode-covering: the model must put mass wherever the data does, including unlikely corners. **Reverse KL** (mode-seeking) appears as the KL penalty in RLHF and in on-policy distillation (Ch 1, 18).

**3.** Repeated peeking inflates the false-positive rate well above 5%: an A/A test checked daily for 20 days reaches "significance" about 25% of the time. Fix: a pre-registered sample size from a power analysis and one look at the end, or a sequential method designed for peeking (alpha spending, always-valid p-values) (Ch 2).

**4.** Standard error of each accuracy ≈ 1.7 points; an unpaired difference has SE ≈ 2.4 points, so 2 points is within noise. Because both models see the same items, use a **paired** test (McNemar, or a paired bootstrap over items), which depends only on items where they disagree and has lower variance. If still unclear, collect more items or look at slices (Ch 2, 23).

**5.** In order: leakage (features from after the prediction time, target leakage, duplicates or the same users across splits, a random split where a time split was needed); training–serving skew (features computed differently online); a label definition that differs from the production outcome; distribution shift since the data was collected; and the metric itself (AUC fine, but the operating threshold or calibration wrong) (Ch 3, 47).

**6.** About ln 10 ≈ 2.30. A loss of 7.5 means the initial logits are far too large or confident: wrong output-layer initialisation, a missing or double softmax, labels misaligned or out of range. Fix the start before tuning anything (Ch 5).

**7.** m ← β₁m + (1 − β₁)g; v ← β₂v + (1 − β₂)g²; bias-correct; θ ← θ − η(m̂/(√v̂ + ε) + λθ). In Adam with L2 the decay term passes through the adaptive scaling, so weights with large gradient variance are barely regularised; AdamW applies decay directly to the weights, so it acts uniformly and is tunable independently of the learning rate (Ch 7).

**8.** Early second-moment estimates are poor and small, so the first steps are large and noisy; warm-up keeps them small until v is reliable. WSD holds the learning rate constant and decays only at the end, so any checkpoint in the stable phase can be decayed into a finished model: you can extend runs and branch experiments without retraining (Ch 7).

**9.** bf16 has fp32's exponent range, so gradients do not overflow or underflow and no loss scaling is needed; fp16 has more mantissa but a narrow range. bf16 has only 8 bits of precision, so small updates to weights near 1 round away; fp32 master weights accumulate them (Ch 7).

**10.** Under SP, each Adam step changes the output roughly in proportion to width, so the learning rate tuned at the smaller width becomes too large and training can diverge; every width needs its own sweep. µP scales initialisation and per-layer learning rates (hidden-matrix learning rates ∝ 1/width) so the best learning rate transfers across widths (Ch 7).

**11.** 2 (K and V) × 32 × 8 × 128 × 8,192 × 2 bytes ≈ **1.07 GB**. With 32 full heads (MHA) it would be 4× larger; GQA shares KV heads across query heads; MLA caches a small latent per token instead of keys and values, a further several-fold cut (Ch 10, 11, 21).

**12.** With unit-variance components, q·k has variance d_k; unscaled scores grow with dimension, the softmax saturates towards one-hot and its gradients vanish. Dividing by √d_k keeps the variance near 1 (Ch 10).

**13.** RoPE rotates pairs of query and key dimensions by angles proportional to position, so q·k depends only on relative distance. Beyond the training length, low-frequency dimensions reach rotation angles never seen in training. Fixes: position interpolation, NTK-aware scaling, YaRN, followed by a short fine-tune on long sequences (Ch 10).

**14.** C ≈ 6ND with D ≈ 20N gives C ≈ 120N², so N ≈ √(1e23/120) ≈ **29B parameters** and D ≈ **580B tokens**. Open models train smaller models far longer because lifetime inference cost dominates: a smaller model trained past compute-optimal is cheaper to serve at the same quality (Ch 13).

**15.** Memory (parameters) scales with the number of experts; compute per token scales with k (2 experts' FFNs). Without balancing, the router collapses onto a few experts, which then overload while others go untrained. Fixes: an auxiliary load-balancing loss (Switch: E · Σ f_e P_e), capacity limits, or auxiliary-loss-free per-expert bias terms (DeepSeek-V3) (Ch 11).

**16.** About 16 bytes per parameter (bf16 weights and gradients, fp32 master weights and two Adam moments): 7e9 × 16 ≈ **112 GB**, more than one 80 GB GPU. ZeRO-3 shards all of it across the 8 GPUs, ≈ 14 GB each, at the cost of gathering weights layer by layer (Ch 17).

**17.** 6ND = 6 × 7e9 × 2e12 = 8.4e22 FLOPs. Throughput 256 × 989e12 × 0.4 ≈ 1.0e17 FLOP/s. Time ≈ 8.3e5 s ≈ **9.6 days**, plus restarts and evaluation (Ch 17).

**18.** Split each document into shingles (word n-grams), hash with many hash functions and keep the minimum per function; the share of matching minima estimates Jaccard similarity. Locality-sensitive hashing (bands) finds candidate pairs without comparing every pair. Duplicates waste compute, raise memorisation and distort the data mix (Ch 16).

**19.** Data parallelism replicates the model and splits the batch, all-reducing gradients (any network, across nodes). Tensor parallelism splits each layer's matrices, all-reducing activations in every layer (needs NVLink, inside a node). Pipeline parallelism splits the layers into stages and sends activations between them (tolerates slower links; costs a pipeline bubble). Large runs combine all three, often with ZeRO/FSDP sharding on the data-parallel axis (Ch 17).

**20.** n-gram or embedding overlap between the training data and the test set; compare performance on the public set against a freshly written or rephrased set of equal difficulty; test items released after the training cutoff; canary strings; check whether the model can complete test questions verbatim. A large gap between original and rephrased items is the strongest evidence (Ch 16, 23).

**21.** ∇J = Σ_y ∇π(y) R(y) = Σ_y π(y) ∇log π(y) R(y) = E[R(y) ∇log π(y)]. For any b independent of y, E[b ∇log π(y)] = b ∇Σ_y π(y) = b ∇1 = 0, so subtracting b changes the variance, not the mean (Ch 18.4).

**22.** The KL-regularised optimum is π*(y) = π_ref(y) exp(r(y)/β)/Z. Solve for r: r = β log(π*/π_ref) + β log Z. Substitute into the Bradley–Terry likelihood; Z cancels in the difference, leaving −log σ(β[log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l))]) (Ch 18.6).

**23.** Mean 0.25, sample standard deviation 0.5: advantages **+1.5, −0.5, −0.5, −0.5** (+1.73 and −0.58 with the population std). If all rewards are equal, every advantage is zero and the prompt gives no gradient. Prompts the policy always or never solves waste generation: filter by pass rate and resample (Ch 19).

**24.** Optimising a learned reward past the point where true quality improves: proxy reward keeps rising while human or gold-model ratings fall (length, flattery, format tricks). Detect with a held-out gold judge or human evaluation plotted against KL from the reference. Limit with a KL penalty or KL budget, early stopping, reward-model ensembles, length control and fresh preference data on the current policy (Ch 18.5).

**25.** 2 × 4,096 × 16 = **131,072**, about 0.8% of the matrix's 16.8M. Fine-tuning updates have low intrinsic rank, so a low-rank update captures most of the change. QLoRA keeps the frozen base in 4-bit (NF4) and trains LoRA adapters in higher precision, so large models fit on one GPU (Ch 20).

**26.** Prefill processes many tokens per weight read (high arithmetic intensity, matrix–matrix work). Decode generates one token per sequence per step, so every step reads all weights and the KV cache for little compute (matrix–vector work). Batching many sequences in decode raises arithmetic intensity, which is why continuous batching and KV-cache memory set serving throughput (Ch 21).

**27.** A small draft model proposes several tokens; the target scores them in one pass; each is accepted with probability min(1, p_target/p_draft), and on rejection a token is resampled from the normalised residual max(0, p_target − p_draft). The result is exactly the target's distribution. It stops helping when acceptance is low (a poor draft) or when the system is already compute-bound at large batch sizes (Ch 21).

**28.** When a single wrong answer is more probable than the right one, voting amplifies it: with 30% correct and one wrong answer taking 70%, majority over 25 samples is right about 2% of the time. Voting also needs answers that can be compared for equality (Ch 19.8).

**29.** Label a sample (a few hundred items) with expert humans; measure judge–human agreement (accuracy or Cohen's kappa) against human–human agreement; check known biases (position, length, self-preference) with swapped and controlled pairs; fix the rubric and prompt version; re-validate when the model or task changes; report judge scores with confidence intervals (Ch 24).

**30.** 0.95²⁰ ≈ **36%**. Errors compound with length, so long tasks need checkpoints, verification and recovery steps; evaluate on end-to-end success and on reliability across repeated runs (pass^k), not single-step accuracy (Ch 25, 33).

**31.** The information is linearly decodable at layer 12, not that the model uses it: probes can pick up correlates. Activation patching swaps that activation between runs that differ only in negation and measures the effect on the output, which tests whether the representation is causally used (Ch 26).

**32.** A head that, at token A, attends to the token after the previous occurrence of A and copies it (… A B … A → B). It needs a previous-token head in an earlier layer (writing "the token before me was A" into each position) and the induction head that matches on that information through K-composition (Ch 26).

**33.** BM25 wins on exact terms, rare names, codes and out-of-domain text; dense retrieval wins on paraphrase and semantic matches. Run both and fuse the ranked lists (reciprocal rank fusion), then re-rank the top candidates with a cross-encoder; evaluate with recall@k on your own queries (Ch 30).

**34.** Check retrieval first: is the right chunk in the top k (recall@k)? If not: chunking split the answer, the query needs rewriting, a metadata filter excluded it, or the embedder misses the vocabulary. If it was retrieved: was it in the context sent to the model (truncation, ordering)? If so, a generation problem: the prompt, conflicting chunks, or the model ignoring context. Fix the earliest failing stage (Ch 31).

**35.** Treat everything from email as data, never instructions; give the agent least privilege (read-only by default, scoped tools); require human confirmation for consequential actions such as sending; restrict where data can flow (no arbitrary URLs, allow-listed recipients); separate the model that reads untrusted content from the one that can act; log and red-team with injection attempts (Ch 22, 28, 33).

**36.** Offline labels are biased by the old ranker's exposure (position bias, never-shown items); the offline metric does not match the online objective; the gain sits in positions or queries that carry little traffic; training–serving skew or latency regressions; too little power to detect the effect; novelty or interference effects. Check skew and power first, then whether offline and online metrics agree historically (Ch 43, 44).

**37.** Scoring every item with a heavy model is too slow at millions of items. Retrieval (two-tower embeddings with approximate nearest-neighbour search, plus other candidate sources) narrows to hundreds or thousands with high recall; a ranker (gradient-boosted trees or a deep cross network using user–item interaction features) orders them precisely; a final re-ranking layer applies business rules and diversity (Ch 43).

**38.** Joining each training label to feature values as they were known at the label's timestamp, not as they are now. Without it, features computed later leak the future into training (for example a restaurant's rating after the order you are predicting), giving inflated offline scores and a model that fails in production (Ch 48).

**39.** A real, consequential mistake owned without blaming others; how you found out you were wrong (data, a review, a user); what you did to fix it and limit the damage; and what you changed afterwards in your process or the team's, with evidence it worked. Mid-level answers pick a trivial mistake or stop at the fix (Ch 38).

**40.** Do not say yes or no to the number: ask what decision the classifier drives and the cost of each error type, which defines the metric (99% accuracy may be trivial at 1% prevalence, or impossible). Check data and label quality, build a quick baseline to see what is achievable, then propose a staged plan: a first version with an agreed precision–recall target and human review of uncertain cases, with a date to revisit (Ch 39).

---

[← E · Worked system designs](E-worked-designs.md) · [Contents](../README.md#contents)
