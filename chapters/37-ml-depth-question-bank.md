# 37 · ML depth question bank

Part X · Research craft and interviews. 160 questions with short model answers, grouped by area and marked by level: **[B]** basic, **[I]** intermediate, **[S]** senior. Each chapter's own interview section goes deeper; this bank is for rapid-fire practice in the last weeks.

## How to use this bank

Every topic in this bank is explained, with links to go deeper, in the **Key concepts** section of the chapter named in each heading. For a second, independent bank of questions, see [Chip Huyen's ML Interviews Book](https://huyenchip.com/ml-interviews-book/).

- **Rapid fire:** read only the question, answer aloud in under a minute, then compare. Mark the ones you fumbled and redo them two days later.
- **Mixed order** in the final week: interviews jump between topics.
- **Go deeper** where an answer feels thin: the chapter number in each section heading points to the full treatment.
- **Senior questions** usually have no single right answer; the model answer shows the structure a strong response follows.

## A. Foundations (Chapters 1–4 and 40)

### Maths and probability

**1. [B] What does cosine similarity measure, and when does it equal the dot product?** The angle between two vectors, ignoring their length. It equals the dot product when both vectors are L2-normalised.

**2. [B] What is the SVD and why does it matter for ML?** Any matrix factors as UΣVᵀ; keeping the top k singular values gives the best rank-k approximation. It underlies PCA, matrix factorisation and the low-rank idea behind LoRA.

**3. [B] State Bayes' rule and give an ML use.** P(A|B) = P(B|A)P(A)/P(B). Naive Bayes classifiers; turning a test's sensitivity and base rate into the probability of fraud given a flag.

**4. [I] MLE vs MAP?** MLE picks parameters maximising the likelihood of the data; MAP adds a prior and maximises the posterior. L2 regularisation is MAP with a Gaussian prior; L1 with a Laplace prior.

**5. [I] Why is cross-entropy the loss for classification?** It is the negative log-likelihood of the labels under the model's predicted distribution, so minimising it is maximum likelihood; it also gives well-behaved gradients (p − y) with softmax or sigmoid.

**6. [I] What is KL divergence and where does it appear in LLM work?** A non-symmetric measure of how one distribution differs from another. It appears in RLHF (penalty keeping the policy close to the reference model), distillation (matching teacher distributions), and variational methods.

**7. [B] Explain the chain rule's role in training neural networks.** Backpropagation applies the chain rule to compute the gradient of the loss with respect to every parameter, layer by layer from the output back.

**8. [I] What is entropy, intuitively?** The average surprise of a distribution: high when outcomes are uncertain, zero when one outcome is certain. Decision trees use it to measure impurity.

### Statistics and experimentation (Chapter 2)

**9. [B] Define a 95% confidence interval correctly.** A procedure that, over repeated samples, produces intervals containing the true value 95% of the time; not a 95% probability for this specific interval.

**10. [B] What is the difference between type I and type II errors?** Type I: a false positive (claiming an effect that is not there), probability α. Type II: a false negative, probability β; power is 1 − β.

**11. [B] How does sample size depend on the minimum detectable effect?** Inversely with its square: halving the MDE needs four times the sample.

**12. [I] What is CUPED?** Variance reduction using a pre-experiment covariate: Y − θ(X − mean X) with θ = cov(Y,X)/var(X); variance falls by 1 − ρ² without bias.

**13. [I] Why is peeking at A/B results a problem?** Repeated significance checks inflate the false positive rate far above α; use a fixed horizon or a sequential method built for interim looks.

**14. [I] What is sample ratio mismatch and what do you do?** The observed traffic split differs significantly from the design; stop interpreting results and find the assignment or logging bug.

**15. [I] Bonferroni vs Benjamini-Hochberg?** Bonferroni controls the chance of any false positive (strict); BH controls the expected share of false discoveries (more power), better for many metrics or segments.

**16. [S] How do you test a change to courier dispatch?** Not by user randomisation, because arms share couriers. Switchbacks over city × time slots, or geo experiments, analysed at the slot level with clustered errors and carryover buffers.

**17. [S] An A/B test shows +4% but leadership doubts it. What do you check?** SRM, data quality, novelty over time, segment consistency, power and the winner's curse, interference, guardrails, and whether the effect replicates in a holdout or follow-up test.

**18. [I] What assumption does difference-in-differences rely on?** Parallel trends: without treatment, treated and control groups would have changed alike; check pre-period trends.

### ML fundamentals (Chapter 3)

**19. [B] Bias vs variance?** Bias: error from too simple a model (underfitting). Variance: error from sensitivity to the training sample (overfitting). Regularisation, data and model choice trade them.

**20. [B] Why can't you use the test set to choose hyperparameters?** It then becomes part of training decisions and no longer gives an unbiased estimate.

**21. [B] L1 or L2 regularisation for feature selection?** L1, because it drives some weights exactly to zero.

**22. [I] Name four types of leakage.** Target-derived features, future information in time-based features, preprocessing fitted on all data, and the same entity across train and test.

**23. [I] How would you cross-validate a model predicting next week's demand?** Walk-forward (time-based) folds with a gap equal to the horizon and point-in-time features.

**24. [I] How does Adam differ from SGD with momentum?** Adam adds per-parameter adaptive step sizes from a running average of squared gradients, with bias correction.

**25. [I] How do you handle a categorical feature with 50,000 values?** Out-of-fold smoothed target encoding, hashing, frequency encoding, or learned embeddings in a neural network; CatBoost handles it natively.

**26. [I] What do you do first with heavy class imbalance?** Choose the right metric (PR-AUC, recall at a precision), then class weights and threshold tuning; resample only training folds; correct probabilities afterwards.

**27. [S] Offline AUC 0.99. Your reaction?** Suspect leakage: inspect top features, timestamps, splits and duplicates; compare with simple baselines.

**28. [S] When should a problem not be solved with ML?** When rules suffice, labels or feedback do not exist, errors are unacceptable and unexplainable, or nobody will act on predictions.

### Classic models (Chapter 40)

**29. [B] How does a decision tree choose a split?** It tries features and thresholds and picks the split with the largest reduction in impurity (Gini, entropy or variance).

**30. [B] Random forest vs gradient boosting?** Forests average deep, decorrelated trees in parallel (variance reduction); boosting adds shallow trees sequentially fitting the gradient (bias reduction), usually more accurate when tuned.

**31. [I] Why does a small learning rate help gradient boosting?** Shrinkage regularises: each tree corrects a little, and with early stopping the ensemble generalises better at the cost of more trees.

**32. [I] Which models need feature scaling?** Linear and logistic regression with regularisation, SVMs, kNN, k-means, PCA and neural networks; trees do not.

**33. [I] What is the kernel trick?** Computing dot products in a high-dimensional feature space via a kernel function, so linear methods learn non-linear boundaries.

**34. [I] How do you choose k in k-means?** Elbow, silhouette, stability, and usefulness for the task; with k-means++ initialisation and restarts.

**35. [S] When would you choose logistic regression over LightGBM in production?** Interpretability or regulation, very small data, critical calibration, extreme latency limits, or when the accuracy gain does not justify complexity.

**36. [S] Why do trees still beat deep learning on most tabular data?** They handle heterogeneous, irregular features and uninformative features well, need little tuning, and deep nets' smoothness bias fits tabular targets poorly; deep models win with huge data, sequences or high-cardinality IDs.

### Evaluation and metrics (Chapter 4)

**37. [B] Precision vs recall?** Precision: share of flagged that are positive. Recall: share of positives that are flagged.

**38. [B] What does ROC-AUC mean?** The probability a random positive scores above a random negative.

**39. [I] Why prefer PR-AUC for rare positives?** ROC-AUC stays high when false positives are many but small relative to the huge negative class; PR-AUC exposes poor precision.

**40. [I] What is calibration and how do you fix it?** Predicted probabilities matching observed rates; fix with Platt scaling, isotonic regression or temperature scaling on held-out data.

**41. [I] MAE, RMSE or WAPE for demand forecasting?** WAPE for aggregate accuracy across items of different volumes; MAE for interpretability; RMSE when large misses are costly; always report bias.

**42. [I] Micro vs macro F1?** Micro pools all decisions (dominated by frequent classes); macro averages per-class F1 (rare classes count equally).

**43. [S] How do you compare two models' AUCs fairly?** Paired bootstrap over the independent unit (users, queries) on the same test set, or DeLong's test; report the interval of the difference.

**44. [S] Offline metric up, online flat: list causes.** Proxy mismatch, different candidate sets, leakage, train-serve skew, feedback-loop bias in logs, low power, or opposing segment effects.

## B. Deep learning, transformers and LLMs (Chapters 5–21)

### Neural networks and PyTorch (Chapters 5–6)

**45. [B] Why do neural networks need non-linear activations?** Without them, stacked layers collapse into one linear function.

**46. [B] Why ReLU (or GELU) instead of sigmoid in hidden layers?** Sigmoid saturates, making gradients vanish in deep networks; ReLU-family activations keep gradients alive and are cheap.

**47. [I] Why does initialisation matter? Name a scheme.** Bad scales make activations and gradients explode or vanish through layers. Xavier/Glorot for tanh, He/Kaiming for ReLU keep variance roughly constant.

**48. [I] Batch norm vs layer norm?** Batch norm normalises each feature across the batch (CNNs, stable large batches); layer norm normalises across features per example (transformers, variable lengths, small batches).

**49. [I] What does dropout do?** Randomly zeroes activations in training, preventing co-adaptation; acts like an ensemble; disabled at inference (with scaling handled in training).

**50. [I] Your loss is NaN after a few steps. What do you check?** Learning rate too high, missing gradient clipping, log of zero or division by zero, fp16 overflow (use bf16 or loss scaling), bad input data.

**51. [I] The model does not learn at all. How do you debug?** Overfit a single batch first; check labels and inputs line up; verify loss and output shapes; check the learning rate; confirm `optimizer.zero_grad()`, `model.train()` and that gradients flow (no detached tensors).

**52. [B] What does** `model.eval()` **change?** Disables dropout and makes batch norm use running statistics; it does not turn off gradients (use `torch.no_grad()` for that).

**53. [I] What is mixed-precision training?** Computing in bf16 or fp16 while keeping critical values (master weights, reductions) in fp32; faster and smaller with near-identical results.

**54. [I] What is a learning-rate warm-up and why use it?** Start with a small learning rate and increase it over the first steps, avoiding unstable early updates when Adam's statistics are poor; standard for transformers.

### Text, embeddings and sequence models (Chapters 8–9)

**55. [B] How does word2vec learn embeddings?** Predicting context words from a word (skip-gram) or the reverse, with negative sampling; words in similar contexts get similar vectors.

**56. [I] Why did LSTMs replace vanilla RNNs, and why did transformers replace LSTMs?** LSTM gates ease vanishing gradients over long sequences; transformers removed sequential computation, allowing parallel training and direct long-range attention.

**57. [B] What is BPE tokenisation?** Start from characters or bytes and repeatedly merge the most frequent pair into a new token, giving a sub-word vocabulary that handles rare words.

**58. [I] Why do tokenisers matter for cost and multilingual quality?** Cost and context are counted in tokens; languages under-represented in the tokeniser's training (often Arabic, among others) need more tokens per word, so they are slower, costlier and often lower quality.

### The transformer (Chapter 10)

**59. [B] Write the attention formula.** softmax(QKᵀ/√dₖ + mask)V.

**60. [B] Why the √dₖ?** Keeps dot products from growing with dimension and saturating softmax.

**61. [I] What does each part of a transformer block do?** Attention mixes information across positions; the feed-forward network transforms each position; residuals and normalisation keep deep training stable.

**62. [I] What is RoPE?** Rotary position embeddings rotate queries and keys by position-dependent angles so attention depends on relative distance.

**63. [I] What is the KV cache?** Stored keys and values of previous tokens per layer so each new token only computes its own; memory grows with layers × KV heads × head dim × tokens.

**64. [I] MQA and GQA?** Query heads share key/value heads (all in MQA, in groups in GQA), shrinking the KV cache and speeding decoding.

**65. [S] Why is attention quadratic, and what are the main ways around it?** Every token attends to every other: O(T²). FlashAttention reduces memory traffic (still exact); sliding-window, sparse or linear attention and state-space models reduce the computation itself.

### Model families and training (Chapters 12, 15)

**66. [B] Encoder vs decoder models: which for what?** Encoders (BERT) for classification, extraction and embeddings; decoders (GPT) for generation and general LLM use.

**67. [I] What is a mixture-of-experts model?** Each token is routed to a few expert feed-forward networks, so total parameters are large but compute per token is small; needs load balancing.

**68. [I] What do scaling laws say?** Loss falls predictably as a power law in parameters, data and compute; Chinchilla showed compute-optimal training uses roughly 20 tokens per parameter, so many earlier models were under-trained.

**69. [B] What are the stages of training a chat LLM?** Pre-training on large text corpora (next-token prediction), supervised fine-tuning on instructions and conversations, then preference optimisation (RLHF or DPO), often followed by reinforcement learning for reasoning.

**70. [I] How does RLHF work?** Collect human preference pairs, train a reward model on them, then optimise the policy with RL (classically PPO) to maximise reward with a KL penalty to stay close to the SFT model.

**71. [I] What does DPO change?** It optimises the policy directly on preference pairs with a classification-style loss derived from the RLHF objective, with no separate reward model or RL loop.

**72. [S] What are reasoning models and how are they trained?** Models that generate long reasoning before answering, trained largely with reinforcement learning on tasks with verifiable rewards (maths, code tests), so they learn to search, check and correct; controlled at inference with a reasoning budget.

**73. [I] What is distillation?** Training a smaller student to imitate a larger teacher's outputs (labels, soft probabilities or generated text), keeping much of the quality at lower cost.

**74. [S] Small fine-tuned model or large general model for a production task?** Compare on your evaluation set with cost and latency: large models for broad, changing or low-volume tasks; small fine-tuned models for narrow, high-volume, latency-sensitive tasks, often distilled from the large one.

### Inference and serving (Chapter 21)

**75. [B] Temperature, top-k and top-p?** Temperature scales logits (lower is more deterministic); top-k samples from the k most likely tokens; top-p from the smallest set whose probability mass reaches p.

**76. [I] Greedy decoding vs beam search for chat models?** Beam search finds high-likelihood sequences but tends to be bland and repetitive for open-ended text; chat uses sampling; beam search suits translation-like tasks.

**77. [I] What is continuous batching?** Adding new requests to the running batch at each decoding step instead of waiting for the whole batch to finish, keeping GPUs busy and raising throughput (vLLM, TGI).

**78. [I] What is speculative decoding?** A small draft model proposes several tokens; the large model verifies them in one forward pass and accepts the matching prefix; same output distribution, lower latency.

**79. [I] What does quantisation trade off?** Lower precision (int8, int4: GPTQ, AWQ) cuts memory and raises speed, with some quality loss that grows at lower bits and for smaller models; measure on your tasks.

**80. [I] Why is decoding memory-bandwidth-bound?** Each step generates one token per sequence but must read all weights and the KV cache; batching amortises the reads.

**81. [I] What are time to first token and tokens per second, and what drives each?** TTFT is dominated by prefill (prompt length, queueing); tokens per second by decode speed (model size, batching, bandwidth, speculative decoding).

**82. [S] Estimate the monthly cost of an LLM feature.** Calls per month × (input tokens × input price + output tokens × output price), minus caching and batch discounts; compare with self-hosting (GPU hours at the needed throughput and utilisation).

**83. [S] Your offline LLM classification job is too expensive. Options?** Batch API (often around half price), prompt caching, shorter prompts, a smaller model or a cascade, and distilling into a small model; a before-and-after cost figure from your own work is the strongest answer.

**84. [S] When would you self-host an open model instead of using an API?** High, steady volume where GPUs stay busy; data residency or privacy requirements; need for fine-tuning or custom decoding; latency control. Against: operational cost, scaling, and keeping up with model releases.

## C. Building with LLMs (Chapters 20–33)

### Prompting and context (Chapter 22)

**85. [B] Zero-shot vs few-shot?** Instructions only vs instructions plus examples; add examples when format or boundary cases are ambiguous.

**86. [B] How do you get reliable JSON?** Structured-output or tool-calling modes with a schema, validation with Pydantic, one retry with the error, and a fallback.

**87. [I] When does chain of thought help?** Multi-step reasoning; not simple extraction. Reasoning models do it internally under a budget.

**88. [I] How do you version and test prompts?** Version template, model and parameters together; run a labelled evaluation set on each change; log versions with outputs.

**89. [I] How do you design a prompt for caching?** Stable content first (system prompt, instructions, examples, tools), byte-identical across calls; per-request content last.

**90. [S] What is indirect prompt injection and how do you defend?** Instructions hidden in content the model reads (documents, emails, tool output). Least privilege, policies in code, human approval for risky actions, untrusted content handled by a tool-less reader, filtering and red-teaming.

### Fine-tuning (Chapter 20)

**91. [B] Prompt, RAG or fine-tune?** Prompt first; RAG for knowledge that changes or must be cited; fine-tune for consistent format or style, narrow high-volume tasks, or domain language.

**92. [B] Explain LoRA in one sentence.** Freeze W and learn a low-rank update BA (scaled by α/r), training well under 1% of parameters and merging afterwards at no latency cost.

**93. [I] Why is B initialised to zero in LoRA?** So the update starts at zero and training begins exactly from the pretrained model.

**94. [I] What makes QLoRA possible?** 4-bit NF4 quantisation of the frozen base, double quantisation and paged optimisers, with bf16 LoRA adapters receiving gradients.

**95. [I] How do you limit catastrophic forgetting?** PEFT, low learning rate, few epochs, replay of general data, and a regression suite.

**96. [S] How would you evaluate whether a fine-tune was worth it?** Against the best prompted baseline on a held-out set, plus format reliability, a general-ability and safety regression suite, and cost and latency; include maintenance cost when the base model changes.

### Embeddings (Chapter 29)

**97. [B] How are sentence embeddings trained?** Contrastively: pairs that should match are pulled together and in-batch negatives pushed apart (InfoNCE).

**98. [I] What does the temperature do in InfoNCE?** Sharpens or softens the softmax over similarities; low temperature focuses on hard negatives.

**99. [I] What are false negatives in hard-negative mining?** Mined "negatives" that are actually relevant; filter them with a cross-encoder or margin, or skip top ranks.

**100. [I] What are Matryoshka embeddings?** Embeddings trained so that prefixes (first 64, 128, … dimensions) are good embeddings too, allowing cheap truncated search.

**101. [S] How do you choose an embedding model?** Shortlist from MTEB for your task and languages, evaluate on your own labelled queries, compare latency, dimension, cost and licence, consider fine-tuning.

### Retrieval (Chapter 30)

**102. [B] BM25 in one sentence.** Keyword scoring summing IDF-weighted, saturating, length-normalised term frequencies.

**103. [B] Why hybrid search?** Keyword search catches exact names and codes; embeddings catch paraphrases; fused (RRF) they cover each other's misses.

**104. [I] How does HNSW work?** A layered proximity graph searched greedily from sparse top layers down to the full bottom layer; efSearch trades recall for latency.

**105. [I] What is product quantisation?** Splitting vectors into sub-vectors, each stored as the id of its nearest of 256 centroids; scored with lookup tables; compressed and approximate.

**106. [I] Bi-encoder vs cross-encoder?** Bi-encoders embed separately (fast, indexable, for retrieval); cross-encoders read query and document together (accurate, slow, for re-ranking).

**107. [S] How would you handle per-user permissions in vector search?** Filter in retrieval (pre-filtering or a partition per tenant), never by prompt instructions; test that restricted documents never reach the context.

### RAG (Chapter 31)

**108. [B] Draw the RAG pipeline.** Offline: load, parse, chunk, embed and index. Online: rewrite query, hybrid retrieval with filters, re-rank, assemble context, generate with citations, verify.

**109. [I] How do you choose chunk size?** Empirically: structure-aware chunks with headings, two or three sizes compared on retrieval recall and answer quality.

**110. [I] What is HyDE?** Embedding an LLM-written hypothetical answer to search with, because it resembles documents more than a short question does.

**111. [I] What is contextual retrieval?** Prepending a short LLM-written description of each chunk's place in its document before embedding and BM25 indexing, so chunks are self-explanatory.

**112. [S] RAG answers are wrong 20% of the time. How do you find the cause?** For a sample of failures, check whether the right passage was retrieved, whether it reached the final context, and whether the answer used it; fix the dominant stage.

**113. [S] When would you use GraphRAG or agentic RAG?** GraphRAG for global or multi-hop questions over a stable entity-rich corpus; agentic RAG for multi-part questions mixing documents and live data. Both cost more; start with a fixed pipeline.

### Agents (Chapter 33)

**114. [B] Workflow or agent?** Workflow when steps are known; agent when the path depends on findings. Prefer the simplest that works.

**115. [I] What makes a good tool for an agent?** Clear description with units and limits, strict schema, concise output, actionable errors, idempotency, and policy checks inside the tool.

**116. [I] Why do multi-step agents fail more often?** Errors compound: per-step success p gives roughly pⁿ over n steps.

**117. [I] What is MCP?** An open protocol exposing tools, resources and prompts from servers to AI applications through a standard interface.

**118. [S] How do you take an agent to production safely?** Sandbox evaluation on real cases over repeated runs, guardrails and budgets, audit logs, shadow mode against humans, gradual automation by risk tier, monitoring of success, cost and overrides.

### Evaluation (Chapters 24–25)

**119. [B] Why are public benchmarks insufficient?** They do not reflect your inputs, criteria or failure costs, and can be contaminated.

**120. [I] Where do you start evaluating a new LLM feature?** Error analysis: read 50–100 traces, write notes, group into failure modes, build an evaluation per frequent mode.

**121. [I] How do you validate an LLM judge?** Against expert labels: Cohen's kappa, true positive and true negative rates, compared with human-human agreement, on a held-out set.

**122. [I] Name three LLM-judge biases.** Position, verbosity, self-preference.

**123. [I] Faithfulness vs correctness?** Faithfulness: claims supported by the retrieved context. Correctness: claims true against a reference. A faithful answer from a wrong document is still wrong.

**124. [S] How many examples do you need to compare two prompts?** Depends on the difference: use paired comparisons on the same cases (McNemar); 100 cases give about ±7 points on a single rate near 85% (±10 near 50%), small differences need hundreds.

**125. [S] How do you evaluate an agent?** End-state task success, trajectory checks, efficiency, robustness to tool failures, repeated runs (pass^k), safety tests.

### Safety and responsible AI (Chapter 28)

**126. [B] What is hallucination, and how do you reduce it in products?** Confident, unsupported output. Ground on retrieved sources with citations, allow abstention, verify claims, constrain outputs, and hand off when unsure.

**127. [I] How do you handle PII in an LLM pipeline?** Minimise what is sent, redact or pseudonymise before calls, contractual and regional controls with providers, no PII in logs or training data without consent, output filters.

**128. [I] How would you moderate user reviews with ML?** Policy definitions first; a fast classifier for clear cases, an LLM for nuanced ones, human review for borderline and appeals; precision and recall per policy category, per language; monitoring for new abuse patterns.

**129. [S] What does the EU AI Act mean for a practitioner?** A risk-based framework: prohibited practices, high-risk systems with obligations (risk management, data governance, documentation, human oversight, accuracy and robustness), transparency duties (for example disclosing AI interaction and AI-generated content), and obligations for general-purpose AI model providers. Know which tier your system falls into and document accordingly.

## D. Recommendation, search, marketplace and systems (Chapters 43–48)

### Recommender systems (Chapter 43)

**130. [B] Why multi-stage recommenders?** Cheap high-recall retrieval over the catalogue, then filtering, then a heavy ranker on hundreds, then page-level re-ranking, to meet latency and cost.

**131. [B] Implicit vs explicit feedback?** Explicit ratings are rare and biased; implicit behaviour (impressions, clicks, orders) is abundant but has no explicit negatives.

**132. [I] How does a two-tower model train and serve?** User and item towers trained with in-batch softmax (plus logQ correction); item vectors precomputed into an ANN index; the user vector computed per request.

**133. [I] What is position bias and how do you correct it?** Items shown higher get more clicks regardless of relevance; use position as a training feature removed at serving, or inverse propensity weighting with propensities from randomisation.

**134. [I] How do you handle cold start for a new restaurant?** Content features in the item tower, ID dropout in training, exploration impressions, borrowing from similar restaurants.

**135. [I] Explain Thompson sampling.** Keep a posterior per option (Beta for click rates); sample once from each; pick the highest; update with the outcome.

**136. [S] Offline NDCG up 3%, A/B flat. Why?** Candidate set mismatch, label vs goal mismatch, leakage, position bias in test labels, re-ranking rules overriding, low power, or segment effects.

**137. [S] Where would you use LLMs in a recommender?** Offline item enrichment and cold-start features, calibrated LLM judges for offline evaluation, conversational surfaces, semantic IDs; not for scoring thousands of candidates per request.

### Search and learning to rank (Chapter 44)

**138. [B] Pointwise, pairwise and listwise?** Per document, per pair, per list; the last two optimise order directly.

**139. [I] Explain LambdaMART.** RankNet's pairwise gradients scaled by the |ΔNDCG| of swapping each pair (LambdaRank), fitted by gradient-boosted trees.

**140. [I] Why must ideal DCG use all relevant documents?** Otherwise a system that retrieves few documents is not penalised for what it missed.

**141. [I] How do you handle a navigational query for a closed restaurant?** Show it first, labelled closed with opening time, then similar open alternatives.

**142. [I] What is team-draft interleaving?** Alternately drafting results from two rankers into one list and crediting clicks to the ranker that supplied them; a sensitive comparison.

**143. [S] Search conversion dropped 2%. Debugging plan?** Instrumentation and traffic mix first, then segment by query type, language, city and platform; zero-result and reformulation spikes; supply changes; recent ranking and index changes; compare with browse conversion.

### Marketplace and logistics (Chapter 45)

**144. [B] Why is seasonal naive a strong forecasting baseline?** Delivery demand has strong weekly patterns; the same hour last week captures them for free.

**145. [I] How do you forecast with stock-outs in the history?** Treat sales during stock-outs as censored: exclude or impute those periods and model unconstrained demand.

**146. [I] What quantile should a staffing forecast target?** The newsvendor quantile q = cost of under-forecasting / (cost of under + cost of over).

**147. [I] Decompose an ETA prediction.** Accept, prep time, courier assignment and travel to the restaurant, wait, travel to the customer, hand-over; pickup is the later of food ready and courier arrival.

**148. [S] How would you rebalance courier supply?** Forecast need per zone, solve a min-cost transport LP from surplus to deficit, turn moves into incentives or suggestions, evaluate with switchbacks.

### System design and MLOps (Chapters 46–47)

**149. [B] Name the eight steps of the ML system design framework.** Clarify, frame, data, features, model, evaluation, serving, monitoring.

**150. [I] What is train-serve skew and how do you prevent it?** Features computed differently in training and serving; prevent with a feature store or shared code paths, logging online features, and comparing with offline recomputations.

**151. [I] Shadow, canary or A/B: when each?** Shadow to validate behaviour and latency without user impact; canary to limit blast radius; A/B to measure effect.

**152. [I] What is data drift vs concept drift?** Data drift: input distribution changes. Concept drift: the relationship between inputs and target changes. Monitor inputs, predictions and delayed labels.

**153. [I] What do you monitor for an LLM feature?** Traces of chained calls, token usage and cost, latency, error and refusal rates, sampled judged quality, user feedback, and drift in input topics.

**154. [S] Back-of-the-envelope: 5M daily users, 4 feed loads each. Peak QPS?** 5M × 4 / 86,400 ≈ 230 average; with a peak factor of 3, about 700 QPS.

**155. [S] How would you roll out a new ranking model safely?** Offline evaluation, shadow traffic for latency and sanity, a small canary with guardrails, an A/B or interleaving test, gradual ramp, monitoring and a rollback switch.

### Data for ML (Chapter 48)

**156. [B] What is a point-in-time join?** Joining each training row only with feature values known at its timestamp (for example `merge_asof` backward), preventing leakage.

**157. [I] How do you sessionise events in SQL?** Flag events whose gap from the previous event (LAG) exceeds a threshold, then take a running sum of the flags per user.

**158. [I] How do you measure label quality?** Inter-annotator agreement (Cohen's or Fleiss' kappa), expert audits of samples, and checking confidently disagreeing model predictions.

**159. [I] When does pandas stop being enough?** When data no longer fits comfortably in memory or jobs take too long: push work to SQL, use Polars or DuckDB, or Spark for distributed processing.

**160. [S] How would you set up data contracts for an ML pipeline?** Agreed schemas, types, freshness and quality rules between producers and consumers, enforced by automated checks that fail the pipeline or alert before bad data reaches training or serving.

---

[← 31 · The ML coding interview](31-the-ml-coding-interview.md) · [Contents](../README.md#contents) · [33 · Behavioural interviews →](33-behavioural-interviews.md)
