# F · Diagnostic self-assessment

Forty-eight questions that tell you where to spend your preparation time. Take it before you start a study plan, and retake it four weeks later with fresh questions (F.10 explains how, because the same questions a second time measure memory, not skill). It takes about 100 minutes.

The questions cover the whole book unevenly on purpose: more on the material senior interviews probe hardest (training, post-training, inference, evaluation), and at least one question each on tokenisation, multimodal models, research craft, coding, drift and forecasting, so that no area of a loop goes untested.

**How to take it.** Closed book, no assistant, a timer at two minutes per question. Answer aloud or in writing, as you would in an interview: the reasoning counts, not only the final number. Then score yourself against the answer key at the end of this appendix:

| Score | Meaning |
|---|---|
| 2 | You gave every point in the key, or an equally good answer, within two minutes |
| 1 | You had the main idea but missed a key point, made an arithmetic slip or needed far more than two minutes |
| 0 | Wrong, blank, or the right words without being able to explain them |

Be strict. A 1 you scored as a 2 is a gap you will meet in the interview instead.

**Turning scores into a plan.** Score each area as a percentage of its maximum (two points per question). Most areas have five questions, so the bands below match 0–4, 5–7 and 8–10 out of 10. F.3 has seven questions (out of 14), and F.8 is scored as two separate sub-areas, so that a strong applied-systems score cannot hide a weak research-craft one.

| Area score | What to do with that area's chapters |
|---|---|
| Below 50% | Study in full: read, run the script, do the practice and the research exercise |
| 50–79% | Read the pitfalls and interview questions, redo the practice for the questions you missed, then drill the matching section of the question bank (Chapter 37) |
| 80% or more | Check the "Done when" box, then move on; revisit only in the final week |

**Partial credit.** Where an answer has several parts, the key ends with a *Partial credit* note saying what earns a 1. When unsure between 1 and 2, ask whether an interviewer would have needed to prompt you for the missing part: if yes, it is a 1.

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

**41.** The same paragraph costs 25 tokens in English and 60 tokens in another language with your tokeniser. Why, and what are the consequences for cost, context and quality? What would you change at the tokeniser level? (Chapter 9)

**42.** Write CLIP's training loss for a batch of N image–text pairs. Why does batch size matter so much, and what does SigLIP change? (Chapter 14)

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

Score this area as two sub-areas: applied ML systems (questions 36, 37, 38, 43 and 44) and research craft, coding and judgement (39, 40 and 45–48).

### Applied ML systems (Chapters 43–48)

**36.** A new ranker improves offline NDCG by 3%, but the A/B test is flat. Give the likely reasons, in order.

**37.** Why do large recommender systems use separate retrieval and ranking stages? What model is typical for each?

**38.** What is a point-in-time join, and what goes wrong in training data without one?

**43.** A feature's production distribution over three bins is 10% / 40% / 50%; at training time it was 30% / 40% / 30%. Compute the population stability index. What do you do next? (Chapter 47)

**44.** A product costs 2 to buy and sells for 5; unsold units are thrown away. Daily demand is roughly normal with mean 100 and standard deviation 20. How many units do you stock, and why not 100? (Chapter 45)

### Research craft, coding and judgement (Chapters 34–39)

**39.** "Tell me about a time you were wrong." What separates a senior answer from a mid-level one?

**40.** A product manager asks for a classifier with 99% accuracy by next month. How do you respond?

**45.** Your new method beats the baseline by 0.8 points on one seed. The baseline's seed-to-seed standard deviation is 0.5 points. What do you conclude, and what do you run next? (Chapter 34)

**46.** You suspect a new pretraining-data filter improves reasoning. Design the smallest experiment that would convince a sceptical colleague. (Chapters 16, 34)

**47.** Implement a numerically stable log-softmax over the rows of a 2-D array in NumPy, and say what goes wrong in the naive version. (Chapter 35)

**48.** A language model's training loss sits at ln(vocabulary size) from step 0 and never moves. Which bugs do you check, in what order, and with what quick test? (Chapters 6, 35)

## F.9 Answer key

The points below are what a full-marks answer contains. The chapter in brackets has the full treatment.

**1.** Bayes: true positives 0.001 × 0.95 = 0.00095; false positives 0.999 × 0.01 = 0.00999; precision 0.00095 / 0.01094 ≈ **8.7%**. At low prevalence, precision is dominated by the false-positive rate; report precision at the real base rate, not on a balanced test set (Ch 4).

**2.** Cross-entropy on data = H(p_data) + KL(p_data ‖ p_model); the entropy is fixed, so training minimises **forward KL**. Forward KL is mode-covering: the model must put mass wherever the data does, including unlikely corners. **Reverse KL** (mode-seeking) appears as the KL penalty in RLHF and in on-policy distillation (Ch 1, 18).

**3.** Repeated peeking inflates the false-positive rate well above 5%: an A/A test checked daily for 20 days reaches "significance" about 25% of the time. Fix: a pre-registered sample size from a power analysis and one look at the end, or a sequential method designed for peeking (alpha spending, always-valid p-values) (Ch 2). *Partial credit:* a 1 for "peeking inflates false positives" without a remedy, or a remedy without saying why peeking is wrong.

**4.** Standard error of each accuracy ≈ 1.7 points; an unpaired difference has SE ≈ 2.4 points, so 2 points is within noise. Because both models see the same items, use a **paired** test (McNemar, or a paired bootstrap over items), which depends only on items where they disagree and has lower variance. If still unclear, collect more items or look at slices (Ch 2, 23). *Partial credit:* a 1 for computing the standard error but not using a paired test, or for "use McNemar" with no numbers.

**5.** In order: leakage (features from after the prediction time, target leakage, duplicates or the same users across splits, a random split where a time split was needed); training–serving skew (features computed differently online); a label definition that differs from the production outcome; distribution shift since the data was collected; and the metric itself (AUC fine, but the operating threshold or calibration wrong) (Ch 3, 47). *Partial credit:* a 1 for naming leakage but not training–serving skew and label mismatch; ordering matters, because leakage is both the most common and the cheapest to check.

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

**18.** Split each document into shingles (word n-grams), hash with many hash functions and keep the minimum per function; the share of matching minima estimates Jaccard similarity. Locality-sensitive hashing (bands) finds candidate pairs without comparing every pair. Duplicates waste compute, raise memorisation and distort the data mix (Ch 16). *Partial credit:* a 1 for the MinHash mechanism without banding (LSH), or without a reason deduplication helps.

**19.** Data parallelism replicates the model and splits the batch, all-reducing gradients (any network, across nodes). Tensor parallelism splits each layer's matrices, all-reducing activations in every layer (needs NVLink, inside a node). Pipeline parallelism splits the layers into stages and sends activations between them (tolerates slower links; costs a pipeline bubble). Large runs combine all three, often with ZeRO/FSDP sharding on the data-parallel axis (Ch 17). *Partial credit:* a 1 if one of the three is wrong about what it communicates or where it runs.

**20.** n-gram or embedding overlap between the training data and the test set; compare performance on the public set against a freshly written or rephrased set of equal difficulty; test items released after the training cutoff; canary strings; check whether the model can complete test questions verbatim. A large gap between original and rephrased items is the strongest evidence (Ch 16, 23). *Partial credit:* a 1 for overlap checks only; a 2 needs a behavioural test (rephrased or post-cutoff items).

**21.** ∇J = Σ_y ∇π(y) R(y) = Σ_y π(y) ∇log π(y) R(y) = E[R(y) ∇log π(y)]. For any b independent of y, E[b ∇log π(y)] = b ∇Σ_y π(y) = b ∇1 = 0, so subtracting b changes the variance, not the mean (Ch 18.4).

**22.** The KL-regularised optimum is π*(y) = π_ref(y) exp(r(y)/β)/Z. Solve for r: r = β log(π*/π_ref) + β log Z. Substitute into the Bradley–Terry likelihood; Z cancels in the difference, leaving −log σ(β[log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l))]) (Ch 18.6).

**23.** Mean 0.25, sample standard deviation 0.5: advantages **+1.5, −0.5, −0.5, −0.5** (+1.73 and −0.58 with the population std). If all rewards are equal, every advantage is zero and the prompt gives no gradient. Prompts the policy always or never solves waste generation: filter by pass rate and resample (Ch 19).

**24.** Optimising a learned reward past the point where true quality improves: proxy reward keeps rising while human or gold-model ratings fall (length, flattery, format tricks). Detect with a held-out gold judge or human evaluation plotted against KL from the reference. Limit with a KL penalty or KL budget, early stopping, reward-model ensembles, length control and fresh preference data on the current policy (Ch 18.5). *Partial credit:* a 1 for the definition and a KL penalty without a way to detect it (a gold judge or human ratings plotted against KL).

**25.** 2 × 4,096 × 16 = **131,072**, about 0.8% of the matrix's 16.8M. Fine-tuning updates have low intrinsic rank, so a low-rank update captures most of the change. QLoRA keeps the frozen base in 4-bit (NF4) and trains LoRA adapters in higher precision, so large models fit on one GPU (Ch 20).

**26.** Prefill processes many tokens per weight read (high arithmetic intensity, matrix–matrix work). Decode generates one token per sequence per step, so every step reads all weights and the KV cache for little compute (matrix–vector work). Batching many sequences in decode raises arithmetic intensity, which is why continuous batching and KV-cache memory set serving throughput (Ch 21).

**27.** A small draft model proposes several tokens; the target scores them in one pass; each is accepted with probability min(1, p_target/p_draft), and on rejection a token is resampled from the normalised residual max(0, p_target − p_draft). The result is exactly the target's distribution. It stops helping when acceptance is low (a poor draft) or when the system is already compute-bound at large batch sizes (Ch 21).

**28.** When a single wrong answer is more probable than the right one, voting amplifies it: with 30% correct and one wrong answer taking 70%, majority over 25 samples is right about 2% of the time. Voting also needs answers that can be compared for equality (Ch 19.8).

**29.** Label a sample (a few hundred items) with expert humans; measure judge–human agreement (accuracy or Cohen's kappa) against human–human agreement; check known biases (position, length, self-preference) with swapped and controlled pairs; fix the rubric and prompt version; re-validate when the model or task changes; report judge scores with confidence intervals (Ch 24). *Partial credit:* a 1 for "compare with human labels" without chance-corrected agreement (kappa) and a human–human ceiling, or without any bias check; a 2 needs agreement against a ceiling, at least two bias controls, and revalidation when the judge or task changes.

**30.** 0.95²⁰ ≈ **36%**. Errors compound with length, so long tasks need checkpoints, verification and recovery steps; evaluate on end-to-end success and on reliability across repeated runs (pass^k), not single-step accuracy (Ch 25, 33).

**31.** The information is linearly decodable at layer 12, not that the model uses it: probes can pick up correlates. Activation patching swaps that activation between runs that differ only in negation and measures the effect on the output, which tests whether the representation is causally used (Ch 26). *Partial credit:* a 1 if you say the probe shows the model uses negation, or describe patching without what it is compared with.

**32.** A head that, at token A, attends to the token after the previous occurrence of A and copies it (… A B … A → B). It needs a previous-token head in an earlier layer (writing "the token before me was A" into each position) and the induction head that matches on that information through K-composition (Ch 26).

**33.** BM25 wins on exact terms, rare names, codes and out-of-domain text; dense retrieval wins on paraphrase and semantic matches. Run both and fuse the ranked lists (reciprocal rank fusion), then re-rank the top candidates with a cross-encoder; evaluate with recall@k on your own queries (Ch 30).

**34.** Check retrieval first: is the right chunk in the top k (recall@k)? If not: chunking split the answer, the query needs rewriting, a metadata filter excluded it, or the embedder misses the vocabulary. If it was retrieved: was it in the context sent to the model (truncation, ordering)? If so, a generation problem: the prompt, conflicting chunks, or the model ignoring context. Fix the earliest failing stage (Ch 31). *Partial credit:* a 1 for checking retrieval only, or for listing causes without the order (retrieval, then context assembly, then generation).

**35.** Treat everything from email as data, never instructions; give the agent least privilege (read-only by default, scoped tools); require human confirmation for consequential actions such as sending; restrict where data can flow (no arbitrary URLs, allow-listed recipients); separate the model that reads untrusted content from the one that can act; log and red-team with injection attempts (Ch 22, 28, 33). *Partial credit:* a 1 for prompt-level defences only ("tell the model to ignore instructions in emails"); a 2 needs at least one architectural control: least privilege, confirmation for sending, or separating the reader from the actor.

**36.** Offline labels are biased by the old ranker's exposure (position bias, never-shown items); the offline metric does not match the online objective; the gain sits in positions or queries that carry little traffic; training–serving skew or latency regressions; too little power to detect the effect; novelty or interference effects. Check skew and power first, then whether offline and online metrics agree historically (Ch 43, 44). *Partial credit:* a 1 for listing reasons without saying which to check first, or for missing exposure (position and selection) bias; a 2 orders the checks, cheapest and most likely first (skew and power), and names offline–online metric mismatch.

**37.** Scoring every item with a heavy model is too slow at millions of items. Retrieval (two-tower embeddings with approximate nearest-neighbour search, plus other candidate sources) narrows to hundreds or thousands with high recall; a ranker (gradient-boosted trees or a deep cross network using user–item interaction features) orders them precisely; a final re-ranking layer applies business rules and diversity (Ch 43). *Partial credit:* a 1 for the funnel without a typical model at each stage, or without the reason (latency over a large catalogue).

**38.** Joining each training label to feature values as they were known at the label's timestamp, not as they are now. Without it, features computed later leak the future into training (for example a restaurant's rating after the order you are predicting), giving inflated offline scores and a model that fails in production (Ch 48).

**39.** A real, consequential mistake owned without blaming others; how you found out you were wrong (data, a review, a user); what you did to fix it and limit the damage; and what you changed afterwards in your process or the team's, with evidence it worked. Mid-level answers pick a trivial mistake or stop at the fix (Ch 38). *Partial credit:* a 1 for a real mistake with a fix but no change in process, or for a story whose "mistake" is really a strength; a 2 needs ownership, how you found out, the fix, and a lasting change with evidence it worked.

**40.** Do not say yes or no to the number: ask what decision the classifier drives and the cost of each error type, which defines the metric (99% accuracy may be trivial at 1% prevalence, or impossible). Check data and label quality, build a quick baseline to see what is achievable, then propose a staged plan: a first version with an agreed precision–recall target and human review of uncertain cases, with a date to revisit (Ch 39). *Partial credit:* a 1 for asking about the decision and costs but not proposing a plan, or for a plan with no baseline or data check; a 2 reframes the metric (prevalence), checks data and labels, offers a staged plan with a review date, and does all of it without simply refusing.


**41.** BPE merges are learned from a training corpus, mostly English, so frequent English words become single tokens while words in under-represented scripts are split into short pieces or bytes: here 60/25 = 2.4 tokens for every English one. Consequences: the same request costs 2.4 times as much and takes longer to generate; the effective context window is 2.4 times shorter; each word takes more prediction steps, which tends to lower quality; and users of that language pay more for the same service. Fixes at the tokeniser level: train it on a language-balanced corpus, enlarge the vocabulary (at the cost of a larger embedding and output layer), and track fertility (tokens per word) per language as a release metric; for an existing model, add vocabulary with new embeddings initialised from the pieces they replace, followed by continued pretraining (Ch 9). *Partial credit:* a 1 for the frequency explanation and one consequence; a 2 needs cost, context and quality, and a concrete fix.

**42.** With unit-norm image and text embeddings, S_ij = cos(image_i, text_j)/τ for a learned temperature τ. The loss is the average of two cross-entropies with the diagonal as the targets: softmax over each row (image to text) and over each column (text to image). Each pair's negatives are the other N − 1 items in the batch, so a larger batch gives more and harder negatives, which is why CLIP used batches of 32,768 and why the N × N matrix must be gathered across devices. SigLIP treats every image–text pair as an independent binary classification with a sigmoid (label 1 on the diagonal, 0 elsewhere, with a learned temperature and bias), so there is no normalisation over the batch: it is cheaper in memory and communication and works well at smaller batch sizes (Ch 14). *Partial credit:* a 1 for one direction of the cross-entropy only, or for the loss without the batch-size argument.

**43.** PSI = Σ (a − e) ln(a/e) = (0.1 − 0.3) ln(1/3) + 0 + (0.5 − 0.3) ln(5/3) = 0.220 + 0.102 ≈ **0.32**, above the 0.25 "investigate" line. First rule out a pipeline change (an upstream schema or unit change, new default values, a broken join), since most sudden shifts are bugs. If the shift is real, check whether the model's predictions and its performance on recently labelled data have moved; retrain or recalibrate only if they have, and compare with the same season last year before calling it drift (Ch 47). *Partial credit:* a 1 for the right number with no plan, or a sound plan with an arithmetic slip.

**44.** Each unit short loses the margin 5 − 2 = 3; each unit over loses its cost, 2. The newsvendor quantile is cost_under / (cost_under + cost_over) = 3/(3 + 2) = 0.6, and the 60th percentile of a normal with mean 100 and standard deviation 20 is 100 + 0.253 × 20 ≈ **105 units**. Not 100, because running out costs more than throwing one away, so the optimum sits above the mean; if unsold units could be returned for most of their cost, it would sit higher still (Ch 45). *Partial credit:* a 1 for the right critical ratio with a slip in the quantile, or for "stock a bit above the mean" without the ratio.

**45.** With one seed per arm, the difference between two runs has standard deviation 0.5 × √2 ≈ 0.71 points, so 0.8 is about one standard deviation: no evidence yet. Run at least five seeds of each, with the baseline tuned as hard as the method: the standard error of the difference then falls to 0.5 × √(2/5) ≈ 0.32, and a true 0.8-point gain would show as about 2.5 standard errors. Then check that the gain holds on a second evaluation and at a second scale, at matched compute (Ch 34). *Partial credit:* a 1 for "run more seeds" without the arithmetic, or for the arithmetic without a plan beyond seeds.

**46.** State the hypothesis and the metric before running: "the filter improves reasoning benchmarks X and Y at matched tokens". Train proxy models (for example 1B parameters on 25B tokens) with and without the filter, identical in everything else, and match the number of training tokens, since a filter that removes data changes how often the rest repeats. Use several seeds to measure the noise floor and evaluations that are above chance at that scale. Add a dose–response check (two or three filter thresholds) and confirm the winner at a second scale. Check that the filter does not simply select text that resembles the benchmarks (contamination by similarity), and report all evaluations, including any that got worse (Ch 16, 34). *Partial credit:* a 1 for a with/without comparison that misses seeds, matched tokens or the contamination check.

**47.** Subtract each row's maximum before exponentiating: `z = x - x.max(axis=1, keepdims=True); return z - np.log(np.exp(z).sum(axis=1, keepdims=True))`. Subtracting a constant from a row leaves the softmax unchanged, and after it the largest exponent is exp(0) = 1, so nothing overflows. The naive version computes exp(x) directly: in float32 anything above about 88.7 overflows to infinity and the result becomes NaN, and very negative logits underflow to 0 so the log gives −∞. Test it: exp of each output row sums to 1, adding 1,000 to a row leaves the output unchanged, and it matches scipy.special.log_softmax (Ch 35). *Partial credit:* a 1 for the max-subtraction trick without explaining why it is valid or without tests.

**48.** ln(vocabulary size) is the correct loss for a uniform prediction at initialisation, so the model is not learning at all. Check, in order: the optimiser does not update the weights (optimizer.step() missing, zero_grad() called after backward, parameters not passed to the optimiser, or the forward pass run under torch.no_grad() or on detached tensors); the learning rate is effectively zero (a warm-up that starts at 0 and lasts the whole run, or a scheduler stepped wrongly); the labels are wrong (not shifted by one for next-token prediction, or padding and masking that leave almost nothing to predict); and the data loader returns the same or empty batch every step. The quick test: try to overfit a single batch for 100 steps, print the gradient norm, and check that one weight tensor actually changes between steps (Ch 6, 35). *Partial credit:* a 1 for two or three plausible bugs without the one-batch overfit test.

## F.10 Retaking the diagnostic

Taking the same 48 questions again after four weeks mostly measures whether you remember the key. For the retake, build a fresh parallel form from the question bank (Chapter 37), whose answers serve as the key. For each area, draw five questions at random from the matching bank sections below, preferring [I] and [S] items, and score them with the same 0/1/2 rule and the same bands.

| Area | Question-bank sections to draw from |
|---|---|
| F.1 Foundations | [Maths and probability](37-ml-depth-question-bank.md#maths-and-probability), [Statistics and experimentation](37-ml-depth-question-bank.md#statistics-and-experimentation-chapter-2), [Evaluation and metrics](37-ml-depth-question-bank.md#evaluation-and-metrics-chapter-4) |
| F.2 Deep learning and training dynamics | [Neural networks and PyTorch](37-ml-depth-question-bank.md#neural-networks-and-pytorch-chapters-56), [Optimisation and training at scale](37-ml-depth-question-bank.md#optimisation-and-training-at-scale-chapters-7-17) |
| F.3 Transformers, architectures and scaling | [The transformer](37-ml-depth-question-bank.md#the-transformer-chapter-10), [Architectures](37-ml-depth-question-bank.md#architectures-chapters-10-11-14), [Text, embeddings and sequence models](37-ml-depth-question-bank.md#text-embeddings-and-sequence-models-chapters-89) |
| F.4 Pretraining | [Pretraining data and scaling](37-ml-depth-question-bank.md#pretraining-data-and-scaling-chapters-13-16), [Optimisation and training at scale](37-ml-depth-question-bank.md#optimisation-and-training-at-scale-chapters-7-17) |
| F.5 Post-training and RL | [Post-training](37-ml-depth-question-bank.md#post-training-chapters-18-20), [Reasoning and RL](37-ml-depth-question-bank.md#reasoning-and-rl-chapters-19-13), [Fine-tuning](37-ml-depth-question-bank.md#fine-tuning-chapter-20) |
| F.6 Inference, prompting and evaluation | [Inference and serving](37-ml-depth-question-bank.md#inference-and-serving-chapter-21), [Prompting and context](37-ml-depth-question-bank.md#prompting-and-context-chapter-22), [Evaluation (Ch 23)](37-ml-depth-question-bank.md#evaluation-chapter-23) |
| F.7 Interpretability, alignment, retrieval and agents | [Interpretability and alignment](37-ml-depth-question-bank.md#interpretability-and-alignment-chapters-26-27), [Retrieval](37-ml-depth-question-bank.md#retrieval-chapter-30), [RAG](37-ml-depth-question-bank.md#rag-chapter-31), [Agents](37-ml-depth-question-bank.md#agents-chapter-33) |
| F.8 Applied ML systems | [Recommender systems](37-ml-depth-question-bank.md#recommender-systems-chapter-43), [Marketplace and logistics](37-ml-depth-question-bank.md#marketplace-and-logistics-chapter-45), [System design and MLOps](37-ml-depth-question-bank.md#system-design-and-mlops-chapters-4647), [Data for ML](37-ml-depth-question-bank.md#data-for-ml-chapter-48) |
| F.8 Research craft, coding and judgement | the [five senior answers](37-ml-depth-question-bank.md#f-five-senior-answers-in-full) and [follow-up chains](37-ml-depth-question-bank.md#g-follow-up-chains), answered aloud before reading; plus one timed exercise from Chapter 35 and one research-design question from Chapter 36 |

Two rules keep the retake honest. Draw the questions before you look at any of them, so you cannot pick favourites. And score an area only against your first-attempt score for the same area, since the bank's questions are of different difficulty from these; a rise of one band is the signal that the study plan worked.
---

[← E · Worked system designs](E-worked-designs.md) · [Contents](../README.md#contents)
