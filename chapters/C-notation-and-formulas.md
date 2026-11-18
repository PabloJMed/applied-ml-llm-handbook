# C · Notation and common formulas

The formulas that come up in interviews, in one place, with the notation used in this handbook. Plain-text maths so it survives export to Markdown and PDF.

## Notation

| Symbol | Meaning |
|---|---|
| x, **x**, X | scalar, vector, matrix |
| x_i, x_t | i-th element / element at time or position t |
| Xᵀ | transpose |
| n, N | number of examples |
| d, d_model, d_k | dimension of features / model / keys |
| V | vocabulary size |
| B, T | batch size, sequence length |
| θ, w, b | parameters, weights, bias |
| ŷ, p̂ | prediction, predicted probability |
| σ(z) | sigmoid, 1 / (1 + e^(−z)) |
| η | learning rate |
| λ | regularisation strength |
| α | significance level (statistics) or acceptance rate (speculative decoding) |
| β | KL / temperature weight in RLHF and DPO |
| E[·], Var(·) | expectation, variance |
| ⊙ | element-wise product |

## Probability and statistics (Ch 1, 2)

- Bayes: P(A|B) = P(B|A) P(A) / P(B)
- Variance: Var(X) = E[X²] − E[X]²; Var(aX + b) = a² Var(X)
- Standard error of a mean: SE = s / √n. Of a proportion: √(p(1 − p) / n)
- 95% confidence interval: estimate ± 1.96 × SE
- Two-sample z for proportions: z = (p₂ − p₁) / √(p(1 − p)(1/n₁ + 1/n₂)), p pooled
- Sample size per arm for a proportion (α = 0.05 two-sided, power 0.8): n ≈ 16 p(1 − p) / δ², δ = absolute effect
- CUPED: Y_adj = Y − θ (X − E[X]), θ = Cov(X, Y) / Var(X); variance falls by a factor (1 − ρ²)
- Delta method for a ratio metric R = Ȳ/X̄: Var(R) ≈ (1/X̄²)[Var(Ȳ) − 2R Cov(Ȳ, X̄) + R² Var(X̄)]
- Entropy: H(p) = −Σ p_i log p_i. Cross-entropy: H(p, q) = −Σ p_i log q_i. KL(p‖q) = Σ p_i log(p_i / q_i)

## Linear models and losses (Ch 1, 3, 4)

- Linear regression: ŷ = Xw + b; closed form w = (XᵀX)⁻¹Xᵀy; ridge adds λI inside the inverse
- MSE = (1/n) Σ (y − ŷ)²; MAE = (1/n) Σ |y − ŷ|
- Logistic regression: p̂ = σ(wᵀx + b); gradient of log-loss w.r.t. w = (1/n) Xᵀ(p̂ − y)
- Binary cross-entropy: −[y log p̂ + (1 − y) log(1 − p̂)]
- Softmax: softmax(z)_i = e^(z_i) / Σ_j e^(z_j). Gradient of cross-entropy w.r.t. logits = p̂ − y (one-hot)
- Hinge loss (SVM): max(0, 1 − y(wᵀx + b)) + (λ/2)‖w‖²
- Gini impurity: 1 − Σ p_k². Entropy for trees: −Σ p_k log₂ p_k
- Gradient boosting step: F_m = F_(m−1) + η h_m, where h_m fits the negative gradient of the loss
- Expected error = bias² + variance + irreducible noise

## Evaluation (Ch 5, 18, 26)

- Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2PR / (P + R); Fβ weights recall β times as much as precision
- FPR = FP / (FP + TN); ROC plots TPR vs FPR; AUC = P(score₊ > score₋)
- Log-loss and Brier score = (1/n) Σ (p̂ − y)² for calibration
- DCG@k = Σ_(i=1..k) (2^(rel_i) − 1) / log₂(i + 1); NDCG@k = DCG@k / IDCG@k
- MRR = mean of 1 / rank of the first relevant item; Recall@k = relevant found in top k / all relevant
- Average precision = mean of precision@k over the ranks k of relevant items; MAP = mean over queries
- Cohen's kappa: κ = (p_o − p_e) / (1 − p_e)
- PSI = Σ (a_i − e_i) ln(a_i / e_i) over bins

## Neural networks (Ch 6, 7)

- Layer: h = f(Wx + b). Backprop: ∂L/∂W = δ xᵀ, δ_prev = Wᵀδ ⊙ f′(z)
- Initialisation: Xavier/Glorot Var(w) = 2 / (fan_in + fan_out); He Var(w) = 2 / fan_in (ReLU)
- SGD with momentum: v ← μv + g; θ ← θ − ηv
- Adam: m ← β₁m + (1 − β₁)g; v ← β₂v + (1 − β₂)g²; bias-corrected m̂, v̂; θ ← θ − η m̂ / (√v̂ + ε). AdamW subtracts ηλθ separately
- Batch norm: x̂ = (x − μ_B) / √(σ²_B + ε), y = γx̂ + β. Layer norm: same over features of one example
- Training memory with AdamW, mixed precision: ≈ 16 bytes per parameter + activations

## Sequence models and attention (Ch 8, 10)

- RNN: h_t = tanh(W_x x_t + W_h h_(t−1) + b)
- LSTM: c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t; h_t = o_t ⊙ tanh(c_t)
- Additive attention score: e_i = vᵀ tanh(W_q s + W_k h_i)
- Scaled dot-product attention: Attention(Q, K, V) = softmax(QKᵀ / √d_k + mask) V
- Multi-head: concat(head_1..head_h) W_O, head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
- Attention cost: O(T² d) time, O(T²) memory for the score matrix (avoided by FlashAttention)
- Parameters per transformer block ≈ 12 d_model² (4d² attention + 8d² MLP with 4× expansion)
- Sinusoidal position: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(…)

## LLM training and alignment (Ch 11, 13)

- Language modelling loss: L = −(1/T) Σ log p(x_t | x_<t); perplexity = e^L
- Training compute: C ≈ 6 N D FLOPs. Chinchilla: D ≈ 20 N, so N ≈ √(C / 120)
- Reward model (Bradley–Terry): L = −log σ(r(x, y_w) − r(x, y_l))
- RLHF objective: max E[r(x, y)] − β KL(π ‖ π_ref)
- DPO: L = −log σ(β [log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)])
- GRPO advantage: A_i = (r_i − mean(r)) / std(r) within a group of samples for one prompt
- Distillation: L = α T² KL(softmax(z_t/T) ‖ softmax(z_s/T)) + (1 − α) CE(y, z_s)
- LoRA: W = W₀ + (α/r) BA, with B ∈ R^(d×r), A ∈ R^(r×k), r ≪ min(d, k); trainable params r(d + k)

## Inference (Ch 14)

- Temperature: p_i ∝ exp(z_i / T)
- KV cache bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_value
- Weights memory = parameters × bytes per parameter (fp16 2, int8 1, int4 0.5)
- Decode speed bound (single stream) ≈ memory bandwidth / bytes read per token
- Speculative decoding: expected tokens per target pass = (1 − α^(k+1)) / (1 − α)
- Latency = TTFT + output_tokens × TPOT
- Cost = requests × (tokens_in × price_in + tokens_out × price_out) / 10⁶

## Retrieval and recommendation (Ch 17, 18, 25)

- Cosine similarity: (a · b) / (‖a‖ ‖b‖); equals the dot product for unit vectors
- TF-IDF: tf(t, d) × log(N / df(t))
- BM25: Σ_t IDF(t) × f(t, d)(k₁ + 1) / (f(t, d) + k₁(1 − b + b |d| / avgdl)), typically k₁ ≈ 1.2–2, b ≈ 0.75
- Reciprocal rank fusion: score(d) = Σ_r 1 / (k + rank_r(d)), k ≈ 60
- InfoNCE: L = −log [exp(sim(q, d⁺)/τ) / Σ_j exp(sim(q, d_j)/τ)]
- Matrix factorisation: r̂_ui = p_u · q_i + b_u + b_i + μ
- BPR loss: −log σ(r̂_ui − r̂_uj) for observed i, unobserved j
- Inverse propensity weighting: unbiased estimate = (1/n) Σ y_i / p(shown_i)
- Thompson sampling (Bernoulli): sample θ_k ~ Beta(1 + successes_k, 1 + failures_k), pick argmax

---

[← B · Cheat sheets](B-cheat-sheets.md) · [Contents](../README.md#contents) · [D · Paper map →](D-paper-map.md)
