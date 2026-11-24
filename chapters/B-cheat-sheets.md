# B · Cheat sheets

One page per area, for the night before. Everything here is a reminder of something explained properly in a chapter; if a line does not make sense, go back to that chapter rather than memorising it.

## B.1 Foundations (Chapters 1–4 and 40)

**Formulas**

| Thing | Formula |
|---|---|
| Standard error of a mean | σ/√n |
| Standard error of a proportion | √(p(1−p)/n); 85% on 100 items → 3.6 points |
| 95% interval | estimate ± 1.96 · SE (Wilson for small-n proportions) |
| Sample size per arm, proportions | (zα/2 + zβ)² [p₁(1−p₁) + p₂(1−p₂)] / (p₂−p₁)²; z = 1.96 and 0.84 for α = 0.05, power 0.8 |
| CUPED | Y − θ(X − X̄), θ = cov(Y,X)/var(X); variance × (1 − ρ²) |
| Bias-variance | error = bias² + variance + noise |
| Logistic gradient | Xᵀ(σ(Xw) − y)/n |
| Adam | m = β₁m + (1−β₁)g; v = β₂v + (1−β₂)g²; w −= η m̂/(√v̂ + ε) |
| Gini impurity | 1 − Σpₖ² |
| Precision; recall; F1 | TP/(TP+FP); TP/(TP+FN); 2PR/(P+R) |
| ROC-AUC | P(score of random positive > random negative) |
| NDCG | Σ (2^rel − 1)/log₂(i+1), divided by ideal DCG over all relevant items |
| WAPE | Σ\|y − ŷ\| / Σ\|y\| |
| Pinball loss at q | max(q(y−ŷ), (q−1)(y−ŷ)) |
| Downsampling correction | p / (p + (1 − p)/w) |
| Kappa | (pₒ − pₑ)/(1 − pₑ) |

**Rules of thumb**

- Halve the MDE, quadruple the sample. Run whole weeks. Check SRM on every test.
- Never peek and stop; never tune on the test set; never split time data randomly.
- Leakage checklist: target-derived features, future data, preprocessing before split, entity overlap, duplicates, overlapping label windows.
- Imbalance: metric first (PR-AUC), then class weights and threshold, resample only training folds, correct probabilities.
- Losses: MSE → mean, MAE → median, pinball → quantile. Newsvendor quantile = cost_under / (cost_under + cost_over).
- Scale features for linear models, SVMs, kNN, k-means, PCA, neural nets; not for trees.
- Tabular default: GBDT (LightGBM/XGBoost/CatBoost) after a linear baseline.
- Bootstrap the independent unit (users, queries), and compare models with paired bootstrap.
- Marketplace experiments: switchbacks or geo when arms share couriers or capacity.

## B.2 Deep learning, transformers and LLMs (Chapters 5–19 and 21)

**Formulas and numbers**

| Thing | Formula or number |
|---|---|
| Attention | softmax(QKᵀ/√dₖ + mask) V |
| Cost per layer | attention O(T²d), FFN O(Td²) |
| KV cache bytes | 2 × layers × KV heads × head dim × tokens × batch × bytes |
| KV cache example | 32 layers, 32 KV heads × 128, 8k tokens, fp16 ≈ 4.3 GB per sequence; 8 KV heads (GQA) ≈ 1.1 GB |
| Training memory (Adam, mixed precision) | ≈ 16 bytes per parameter + activations (7B → ≈ 112 GB) |
| Inference weights | 2 bytes/param in bf16, 1 in int8, 0.5 in int4 (7B → 14 / 7 / 3.5 GB) |
| Chinchilla | compute-optimal ≈ 20 training tokens per parameter (minimises training compute; deployed models train much longer, Ch 13) |
| LoRA parameters per matrix | r(d + k) instead of d·k; update (α/r)BA, B = 0 at start |
| InfoNCE | −log [exp(sᵢᵢ/τ) / Σⱼ exp(sᵢⱼ/τ)] |
| LLM cost | calls × (in tokens × in price + out tokens × out price) − caching − batch discounts |

**Rules of thumb**

- Pre-norm, RMSNorm, SwiGLU, RoPE, GQA: the modern decoder recipe.
- Prefill is compute-bound; decode is memory-bandwidth-bound; batching (continuous) raises throughput.
- Latency levers: smaller model, quantisation, speculative decoding, shorter prompts, caching, streaming.
- Training pipeline of a chat model: pre-training → SFT → preference optimisation (RLHF/DPO) → RL for reasoning.
- Debugging a net: overfit one batch first; NaNs → learning rate, clipping, fp16, bad data.
- Tokenisers decide cost and multilingual quality; Arabic often needs more tokens per word.

## B.3 Retrieval, RAG, building with LLMs, evaluation and safety (Chapters 20 and 22–33)

**Retrieval and RAG**

- BM25 (k₁ ≈ 1.2, b ≈ 0.75) + dense bi-encoder, fused with RRF (Σ 1/(60 + rank)); cross-encoder re-ranks top 50–100.
- ANN: HNSW (M, efConstruction, efSearch) for in-RAM low latency; IVF (n_lists, n_probe); PQ for 10–30× compression with exact re-rank. Always measure ANN recall vs exact.
- Memory: N × d × 4 bytes in float32 (10M × 768 ≈ 31 GB); int8 = ¼; binary = 1/32 plus re-scoring.
- Filters (country, audience, permissions) in retrieval; pre-filter or partition when selective.
- Chunking: structure-aware, headings prepended, 2–3 sizes compared on recall@k.
- Query transforms: conversational rewrite (must), multi-query, decomposition, HyDE; each measured.
- Generation: answer only from context, cite ids, abstain when absent, verify citations.
- Debug order: retrieved? → in final context? → used correctly?

**Prompting, fine-tuning, agents**

- Prompt anatomy: system → instructions/definitions → examples/tools (stable, cacheable) → context → history → user message.
- Structured outputs + Pydantic validation + one retry.
- Prompt → RAG → fine-tune, in that order of reach. LoRA: r 8–64, α ≈ 2r, lr ≈ 1e-4 to 3e-4, all linear layers. QLoRA: NF4 + double quantisation + paged optimisers.
- Workflows before agents; tools with strict schemas, policy in code, budgets, shadow mode. Reliability ≈ pⁿ.
- Injection: never let the model alone stand between untrusted text and a consequential action.

**Evaluation and safety**

- Start with error analysis (read 50–100 traces, group failure modes), then code checks, then calibrated binary judges with critiques.
- Judge calibration: kappa, TPR, TNR vs human labels; compare with human-human agreement. Biases: position, verbosity, self-preference.
- Intervals: 100 cases ≈ ±7 points near 85%, ±10 near 50%; compare systems paired (McNemar) on the same cases.
- RAG metrics: context recall/precision (at final context), faithfulness (claim-level), answer relevance, correctness, abstention (wrongly answered rate).
- Agents: end-state success, trajectory rules, efficiency, robustness, pass^k over repeated runs, safety tests.
- Safety: grounding and abstention against hallucination; PII minimisation and redaction; moderation with policy definitions, per-language metrics, human appeals; EU AI Act risk tiers and transparency duties.

## B.4 Recommendation, search and marketplace (Chapters 43–45)

- Funnel: candidate generation (two-tower ANN, order again, item-to-item, popular nearby) → filtering (open, in range, in stock) → ranking (GBDT/DCN/multi-task) → re-ranking (MMR diversity, rules, ads, exploration).
- Two-tower: in-batch softmax, logQ correction, hard negatives in moderation; cross features belong in the ranker.
- Score = p(click) × p(order|click) × value, with calibrated heads (ESMM for selection bias).
- Position bias: position feature dropped at serving, or IPS with clipped weights; exploration for never-shown items.
- Cold start: content features, ID dropout, exploration slots.
- Evaluation: recall@k for retrieval, NDCG for ranking, no sampled negatives, time splits; online orders per user with guardrails; interleaving for rankers.
- Search: query understanding → hybrid retrieval → LambdaMART on query-document features → blend. Navigational queries pinned; zero-result rate tracked; Arabic/English/transliteration handled.
- LambdaMART = RankNet gradients × |ΔNDCG|, fitted with boosted trees.
- Forecasting: seasonal naive baseline, global GBDT with lags ≥ horizon, quantiles checked for coverage, censored demand handled, hierarchical reconciliation.
- ETA: decompose (prep, courier, travel), promise a quantile, measure late rate and calibration.
- Optimisation: Hungarian for assignment, LP for rebalancing; predict-then-optimise.

## B.5 Systems (Chapters 46–48)

- Framework: clarify → frame → data → features → model → evaluation → serving → monitoring; about 45 minutes, spend the most on the middle four.
- QPS = daily users × requests per user / 86,400, × peak factor (≈ 3). 10M users × 5 requests ≈ 580 QPS average, ≈ 1,700 at peak.
- Latency budget: split the p99 target across stages before choosing models (for example retrieval 30 ms, features 30 ms, ranking 50 ms, re-ranking 10 ms) and keep about a third as headroom for tail effects.
- Little's law: requests in flight = arrival rate × time in system. 1,700 QPS × 150 ms ≈ 255 concurrent requests to provision for.
- Deployment: shadow → canary (1–5% of traffic, automatic rollback on a pre-agreed one-sided test of error rate or guardrails against production) → A/B → ramp; a rollback switch at every stage.
- Drift: PSI = Σ (aᵢ − eᵢ) ln(aᵢ/eᵢ) over bins; below 0.1 stable, 0.1–0.25 watch, above 0.25 investigate. Alert on effect size, not on p-values, which flag harmless shifts at large n.
- Monitor: input drift, prediction drift, delayed-label performance, latency, cost; for LLMs also traces, tokens, sampled judged quality.
- Staffing (couriers, agents, servers): square-root staffing c ≈ a + β√a for offered load a; the safety margin shrinks relative to load as zones are pooled.
- Stocking or staffing under uncertainty: newsvendor quantile = cost_under / (cost_under + cost_over); under-staffing at 4× the cost of idle time → plan to the 80th percentile.
- Point-in-time joins (`merge_asof` backward); data contracts; label agreement (kappa).

## B.6 LLM research (Chapters 7, 11, 13–19, 23, 26–27, 34)

**Numbers to know**

| Thing | Number or formula |
|---|---|
| Training FLOPs | 6ND; time = 6ND / (GPUs × peak × MFU); MFU 35–50% dense |
| Training memory | 16 bytes/param (Adam, mixed precision) + activations; ZeRO-3 divides the 16 bytes by the DP degree (activations are not sharded) |
| Compute-optimal | N_opt ∝ C^(β/(α+β)); about 20 tokens/param minimises training compute; deployed small models train on hundreds to thousands |
| Repeated data | ≈ fresh up to ~4 epochs; little value beyond ~16 |
| Pipeline bubble | (p − 1)/(m + p − 1) |
| Ring all-reduce | each GPU sends ≈ 2× the data, independent of GPU count |
| bf16 / fp16 / e4m3 | ε = 0.0078 / max 65,504 / max 448 |
| MLA cache | (latent + rope dims) × 2 bytes per token per layer: 1,152 vs 65,536 for 128-head MHA |
| MoE balance loss | E · Σ f_e P_e = 1 at uniform routing |
| GRPO advantage | (r − mean)/std per group; token-level mean loss; zero-variance groups teach nothing |
| Zero-signal groups | P(all G rewards equal) = pᴳ + (1 − p)ᴳ; G = 8 at pass rate 0.9 → 43% of groups are wasted |
| Best-of-n KL | ≤ log n − (n − 1)/n (an upper bound): n = 16 → ≤ 1.84 nats |
| pass@k (unbiased) | 1 − C(n−c, k)/C(n, k) |
| Eval SE | √(p(1−p)/n); paired, clustered; ~2,000 items to detect 2 points at 10% disagreement |
| Absence of evidence | 0/n successes → 95% upper bound ≈ 3/n |

**Post-training objectives** (write them from memory; Chapter 18 derives the first four, Chapter 19 the GRPO family)

| Objective | Formula |
|---|---|
| Reward model (Bradley–Terry) | L = −log σ(r(x, y_w) − r(x, y_l)) |
| KL-regularised RL | max E[r(x, y)] − β KL(π ‖ π_ref); optimum π* ∝ π_ref · exp(r/β) |
| Policy gradient (REINFORCE) | ∇J = E[(R − b) ∇log π(y\|x)]; any b that does not depend on y keeps it unbiased |
| PPO-clip (per token) | min(ρA, clip(ρ, 1 − ε, 1 + ε) A), ρ = π_θ/π_old, ε ≈ 0.2 |
| DPO | −log σ(β[log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)]) |
| DPO gradient | −β σ(r̂_l − r̂_w) [∇log π(y_w) − ∇log π(y_l)], r̂ = β log(π/π_ref): pairs already ranked right by a margin of 1 get weight σ(−1) = 0.27, pairs ranked wrong by 1 get 0.73 |
| SimPO | −log σ((β/\|y_w\|) log π(y_w) − (β/\|y_l\|) log π(y_l) − γ); no reference model |
| GRPO | (1/G) Σᵢ (1/\|oᵢ\|) Σₜ min(ρᵢₜ Âᵢ, clip(ρᵢₜ, 1 − ε, 1 + ε) Âᵢ) − β KL_k3, maximised; the KL sits in the loss, not the reward |
| GSPO ratio | sᵢ = (π_θ(oᵢ)/π_old(oᵢ))^(1/\|oᵢ\|), clipped once per response |
| KL estimator k3 | r − 1 − log r per token, r = π_ref/π_θ; unbiased for KL(π_θ ‖ π_ref) and never negative |

**Rules of thumb**

- Tune baselines as hard as methods; compare at matched tokens and decayed checkpoints.
- Run 2–3 seeds of the baseline to know the noise floor; re-run sweep winners.
- Small-scale results are hypotheses until they hold at two scales.
- Read 20 items of any benchmark; run a blind or perturbed baseline; check contamination.
- RL: verify the verifier; watch entropy, length and pass@k, not only pass@1.
- Interpretability: descriptive tools generate hypotheses; interventions test them.
- Safety evals: elicit hard, report intervals, never treat zero observed successes as zero risk.

## B.7 Interview day

- **Coding:** clarify → example → plan aloud → code → test edge cases → complexity and scale. Practise with autocomplete and AI assistants switched off, because most coding rounds still forbid them; if the company says its round allows an assistant, practise that format too, since it then tests how you review and test generated code.
- **ML depth:** answer the question asked in one sentence first, then the detail; say "it depends on…" only followed by the deciding factors.
- **System design:** write the framework headings first; state assumptions with numbers; baseline before fancy; evaluation and monitoring always.
- **Behavioural:** STAR, "I" not "we", numbers, a lesson; five stories ready; positive reasons for the move.
- **Your questions:** success in six months; how GenAI work is prioritised; how LLM features are evaluated; the path to the next level.
- Sleep, water, a printed copy of the framework, and a five-minute review of your error log.

## B.8 Inference numbers (Chapters 10, 21)

The numbers a serving or "size this deployment" question needs. Hardware figures are vendor peak specifications for the SXM parts; plan with 50–70% of them.

| Thing | Number or formula |
|---|---|
| H100 (80 GB) | 3.35 TB/s HBM bandwidth; ≈ 989 TFLOP/s dense bf16, ≈ 1,979 dense fp8 |
| H200 (141 GB) | 4.8 TB/s; the same compute as an H100, so it helps decode (bandwidth-bound) more than prefill |
| Ridge point of the roofline | peak FLOP/s ÷ bandwidth = 989e12 / 3.35e12 ≈ 295 FLOPs per byte; below it an operation is memory-bound |
| Inference FLOPs | ≈ 2N per token for N active parameters, plus attention over the context |
| Decode arithmetic intensity | ≈ b FLOPs per byte for bf16 weights at batch b (2 FLOPs per parameter per sequence, 2 bytes per parameter read once), so decode is memory-bound until b approaches ≈ 300 |
| Single-stream decode bound | bandwidth ÷ bytes read per token: 8B in bf16 on one H100 → 3.35e12 / 16e9 ≈ 209 tokens/s at best |
| Prefill time | 2N × prompt tokens ÷ (peak × MFU): 70B, 2,000 tokens, 8 × H100 at 50% → ≈ 71 ms |
| KV cache per token | 2 × layers × KV heads × head dim × bytes: Llama-3-8B shape (32, 8, 128) in bf16 = 131 KB; Llama-3-70B shape (80, 8, 128) = 328 KB |
| Maximum batch | (GPU memory − weights − ≈ 10% overhead) ÷ KV bytes per sequence |
| Latency | TTFT (queue + prefill) + output tokens × TPOT; 300 tokens at 20 ms ≈ 6 s, so stream |
| Concurrency (Little's law) | sequences in flight = request rate × (TTFT + n_out × TPOT) |
| Speculative decoding | tokens per target pass = (1 − α^(k+1))/(1 − α): α = 0.8, k = 4 → 3.36, before paying for the draft |
| Self-hosted cost per 1M output tokens | GPU $/h × GPUs ÷ (tokens/s × 3,600) × 10⁶: 8 GPUs at an assumed $2.50/h serving 17,000 tokens/s → $0.33 at full use, $0.82 at 40% utilisation |
| Weights | params × bytes: 70B → 140 GB bf16, 70 GB fp8, 35 GB int4 |

**Rules of thumb**

- Prefill is compute-bound, decode bandwidth-bound: size prefill capacity by FLOPs and decode capacity by bytes. Long prompts can make prefill the larger bill (Appendix E.11).
- Quantising weights to 8 or 4 bits speeds decode almost in proportion at small batch; fp8 also doubles tensor-core throughput for prefill. Check quality with a paired evaluation, not a single benchmark.
- Utilisation, not list price, decides self-host versus API: a GPU at 30% load costs three times as much per token as at 90%.
- Report p50 and p99 TTFT and TPOT separately; averages hide queueing, which grows sharply as load approaches capacity.

---

[← A · Glossary](A-glossary.md) · [Contents](../README.md#contents) · [C · Notation and common formulas →](C-notation-and-formulas.md)
