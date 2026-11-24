# C · Notation and common formulas

The formulas that come up in interviews, in one place, with the notation used in this handbook. Most are plain-text maths so they survive export to Markdown and PDF; the post-training objectives and the modern transformer block, which interviewers ask you to write out in full, are displayed as equations with every symbol defined and a small example you can check by hand. Symbols that mean different things in different areas (G, Σ, a, b, h) are defined where they are used.

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
| β | KL / temperature weight in RLHF and DPO; also the data exponent in scaling laws |
| π_θ, π_old, π_ref | the policy being trained, the policy that sampled the data, the frozen reference (usually the SFT model) |
| ρ | importance ratio π_θ/π_old for one token |
| A, Â | advantage, and its estimate |
| G | group size in GRPO (elsewhere: the true gradient) |
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

## Linear models and losses (Ch 1, 3, 40)

- Linear regression: ŷ = Xw + b; closed form w = (XᵀX)⁻¹Xᵀy; ridge adds λI inside the inverse
- MSE = (1/n) Σ (y − ŷ)²; MAE = (1/n) Σ |y − ŷ|
- Logistic regression: p̂ = σ(wᵀx + b); gradient of log-loss w.r.t. w = (1/n) Xᵀ(p̂ − y)
- Binary cross-entropy: −[y log p̂ + (1 − y) log(1 − p̂)]
- Softmax: softmax(z)_i = e^(z_i) / Σ_j e^(z_j). Gradient of cross-entropy w.r.t. logits = p̂ − y (one-hot)
- Hinge loss (SVM): max(0, 1 − y(wᵀx + b)) + (λ/2)‖w‖²
- Gini impurity: 1 − Σ p_k². Entropy for trees: −Σ p_k log₂ p_k
- Gradient boosting step: F_m = F_(m−1) + η h_m, where h_m fits the negative gradient of the loss
- Expected error = bias² + variance + irreducible noise

## Evaluation (Ch 4, 30, 44)

- Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2PR / (P + R); Fβ weights recall β times as much as precision
- FPR = FP / (FP + TN); ROC plots TPR vs FPR; AUC = P(score₊ > score₋)
- Log-loss and Brier score = (1/n) Σ (p̂ − y)² for calibration
- DCG@k = Σ_(i=1..k) (2^(rel_i) − 1) / log₂(i + 1); NDCG@k = DCG@k / IDCG@k
- MRR = mean of 1 / rank of the first relevant item; Recall@k = relevant found in top k / all relevant
- Average precision = mean of precision@k over the ranks k of relevant items; MAP = mean over queries
- Cohen's kappa: κ = (p_o − p_e) / (1 − p_e)
- PSI = Σ (a_i − e_i) ln(a_i / e_i) over bins

## Neural networks (Ch 5, 6)

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
- Parameters per transformer block ≈ 12 d_model² (4d² attention + 8d² MLP with 4× expansion). This holds for multi-head attention with a 4× GELU MLP. In general it is (2 + 2·n_kv/n_heads) d² for attention plus 3·d·d_ff for a SwiGLU MLP: for a Llama-3-8B-shaped block (n_kv/n_heads = 8/32, d_ff = 3.5d) that is 2.5d² + 10.5d² = 13d²
- Sinusoidal position: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(…)

## The modern decoder block (Ch 10, 11)

Most open LLMs since 2023 use the same block: pre-norm with RMSNorm, rotary position embeddings, grouped-query attention and a SwiGLU feed-forward layer.

**RMSNorm.** Layer norm without the mean subtraction and the bias:

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^{2} + \epsilon}} \odot g
$$

where x is one token's d-dimensional activation and g a learned gain. For x = (3, 4), the RMS is √12.5 = 3.54 and the output is (0.85, 1.13) before the gain. It is cheaper than layer norm and works as well, because the re-scaling, not the re-centring, is what stabilises training.

**SwiGLU.** A gated feed-forward layer with three matrices:

$$
\mathrm{FFN}(x) = \left(\mathrm{SiLU}(xW_1) \odot xW_3\right)W_2,\quad \mathrm{SiLU}(z) = z\,\sigma(z)
$$

with W_1, W_3 of shape d × d_ff and W_2 of shape d_ff × d, so 3·d·d_ff parameters. Matching a 4× GELU MLP's 8d² needs d_ff = 8d/3: for d = 4,096 that is 10,923, which Llama-2-7B rounds to 11,008. Many newer models use a wider ratio (Llama-3-8B uses 14,336 = 3.5d).

**RoPE.** Split each query and key into d/2 pairs and rotate pair i of the vector at position m by the angle mθ_i, with θ_i = b^(−2i/d) and base b = 10,000 (much larger in long-context models). Writing pair i as a complex number q^(i):

$$
\tilde{q}^{(i)}_m = q^{(i)}\,e^{\mathrm{i}\,m\theta_i},\quad \mathrm{Re}\left[\tilde{q}^{(i)}_m\,\overline{\tilde{k}^{(i)}_n}\right] = \mathrm{Re}\left[q^{(i)}\,\overline{k^{(i)}}\,e^{\mathrm{i}(m-n)\theta_i}\right]
$$

so every attention score depends only on the offset m − n. With d = 128 the fastest pair turns 1 radian per token (a period of 6.3 tokens) and the slowest 1.15e-4 radians (a period of about 54,000 tokens); context extension methods such as YaRN rescale the slow pairs, whose angles beyond the training length were never seen.

**GQA.** With n_h query heads and n_kv key-value heads of size d_head, the K and V projections have 2·d·n_kv·d_head parameters per layer and the cache stores 2·n_kv·d_head values per token per layer. For d = 4,096, 32 query heads and 8 KV heads of 128, that is 8.4M parameters instead of 33.6M for full multi-head attention, and a cache 4× smaller. MQA is the n_kv = 1 extreme.

## LLM training and alignment (Ch 13, 15, 18)

- Language modelling loss: L = −(1/T) Σ log p(x_t | x_<t); perplexity = e^L
- Training compute: C ≈ 6 N D FLOPs. Chinchilla: D ≈ 20 N, so N ≈ √(C / 120)
- Reward model (Bradley–Terry): L = −log σ(r(x, y_w) − r(x, y_l))
- RLHF objective: max E[r(x, y)] − β KL(π ‖ π_ref)
- DPO: L = −log σ(β [log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)])
- GRPO advantage: A_i = (r_i − mean(r)) / std(r) within a group of samples for one prompt
- Distillation: L = α T² KL(softmax(z_t/T) ‖ softmax(z_s/T)) + (1 − α) CE(y, z_s)
- LoRA: W = W₀ + (α/r) BA, with B ∈ R^(d×r), A ∈ R^(r×k), r ≪ min(d, k); trainable params r(d + k)

## Training dynamics, scaling and systems (Ch 7, 13, 17)

- AdamW in full (the short form is under Neural networks): p ← p(1 − ηλ) − η m̂ / (√v̂ + ε), with m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ) at step t; the decay ηλp is applied to the weights directly, outside the adaptive scaling
- Gradient noise scale: B_simple = tr(Σ) / |G|², where Σ is the covariance of per-example gradients and G the full-batch (true) gradient. Batches much smaller than B_simple waste steps on noise; much larger ones waste compute
- z-loss: λ_z · (log Σ_j e^{z_j})²
- Chinchilla parametric loss: L(N, D) = E + A/N^α + B/D^β, with N parameters, D training tokens, E the irreducible loss of the data and A, B, α, β fitted constants; compute-optimal N_opt = G (C/6)^(β/(α+β)), where here G = (αA/βB)^(1/(α+β)) is a constant, not a gradient
- Chinchilla's "20 tokens per parameter" minimises training compute only; inference-aware optima are much larger
- Effective repeated data (Muennighoff et al.): D' = U + U R* (1 − e^(−R/R*)), with U unique tokens, R = epochs − 1 repetitions and R* ≈ 15.4 the fitted decay constant (how many repeats it takes for their value to fade). Four epochs give D' ≈ 3.73U, worth 93% of fresh data
- MFU = tokens/s × (6N + 12 · layers · d · seq) / peak FLOP/s; the second term counts attention-score FLOPs over a sequence of length seq, and recomputed activations are not counted (that would be hardware FLOPs utilisation)
- Activation memory per layer ≈ s · b · h · (34 + 5 a s / h) bytes in 16-bit with no recomputation, where s is sequence length, b micro-batch size, h hidden size and a the number of attention heads; the 5as/h term is the stored attention matrices, which FlashAttention removes
- Ring all-reduce traffic per GPU: 2 (n − 1)/n × message size
- Pipeline bubble: (p − 1)/(m + p − 1)
- MoE load-balancing loss: E · Σ_e f_e P_e, with E experts, f_e the fraction of tokens routed to expert e (not differentiable) and P_e the mean router probability for expert e (differentiable, so the gradient flows through it); it equals 1 under uniform routing
- MoE parameters: total = shared + E × per-expert parameters, active per token = shared + k × per-expert (top-k routing). Compute and the 6ND rule use active parameters, memory uses total: DeepSeek-V3 has 671B total and 37B active

## Post-training and reasoning (Ch 18, 19)

The objectives below are the ones interviewers ask you to write out and then question term by term. Every response y (or o) is a sequence of tokens; log π(y | x) is the sum of the token log-probabilities.

### PPO

PPO maximises a clipped surrogate per token. With ρ_t = π_θ(a_t | s_t)/π_old(a_t | s_t) the probability ratio of the token under the current and the sampling policy, Â_t its advantage (from GAE) and ε ≈ 0.2:

$$
L^{\mathrm{clip}}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\right)\right]
$$

The min makes the objective pessimistic: once a token's ratio has moved past 1 ± ε in the direction the advantage favours, its gradient is zero, so one batch cannot move the policy far. The full loss that is minimised adds a value-function term and an entropy bonus, L = −L^clip + c_v (V_φ(s_t) − R̂_t)² − c_e H[π_θ(· | s_t)], where V_φ is the value model, R̂_t the return target and H the entropy; c_v ≈ 0.5 and c_e small or zero for language models. In RLHF the KL to the reference model is usually folded into the per-token reward, r_t = −β log(π_θ/π_ref) plus the reward-model score on the last token.

### GRPO

GRPO keeps PPO's clipped ratio but replaces the value model with a group baseline. For a prompt q, sample G responses o_1 … o_G, score them, and give every token of response i the advantage Â_i = (r_i − mean(r)) / std(r). The per-token term is

$$
\ell_{i,t} = \min\left(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\,\hat{A}_i\right) - \beta\, k_{3}(i,t)
$$

with the token ratio

$$
\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t} \mid q, o_{i,<t})}
$$

and the per-token KL estimate k3 = r − 1 − log r with r = π_ref(o_{i,t} | …)/π_θ(o_{i,t} | …). The original objective averages over the tokens of each response, then over responses:

$$
J_{\mathrm{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}
$$

Three points interviewers probe. First, the KL sits in the loss as a separate term, not in the reward as in PPO-RLHF, so it does not pass through the advantage normalisation. Second, the 1/|o_i| factor gives each token of response i the weight Â_i/(G|o_i|), which is the source of the length bias. Third, many recent recipes (DAPO, and others for long-reasoning RL) set β = 0.

**Worked example.** Rewards [1, 0, 0, 0] with lengths [200, 1,000, 1,500, 2,000] tokens. The mean is 0.25 and the sample standard deviation 0.5, so Â = [+1.5, −0.5, −0.5, −0.5]. Under the per-response average, each token of the correct answer gets weight +1.5/(4 × 200) = +1.9e-3, and each token of the wrong answers gets −1.3e-4, −0.83e-4 and −0.63e-4: the longest wrong answer is penalised least per token, so being wrong at length is cheap. DAPO's token-level loss divides by the total token count instead, J = Σ_i Σ_t ℓ_{i,t} / Σ_i |o_i|, giving +1.5/4,700 = +3.2e-4 to every correct token and −1.1e-4 to every wrong one, whatever the length. Dr. GRPO divides by a constant (the maximum length) and also drops the std in Â.

### GSPO and CISPO

GSPO replaces the token ratios by one length-normalised sequence ratio, clipped once per response:

$$
s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\mathrm{old}}(o_i \mid q)}\right)^{1/|o_i|} = \exp\left(\frac{1}{|o_i|}\sum_{t}\log\rho_{i,t}\right)
$$

and maximises (1/G) Σ_i min(s_i Â_i, clip(s_i, 1 − ε, 1 + ε) Â_i), with a much smaller ε than token-level clipping because s_i is an average. Why normalise: 1,000 tokens that each drift by a ratio of 1.001 give a raw sequence ratio of 1.001^1000 = 2.72, outside any sensible clip range, while the geometric mean stays at 1.001. CISPO keeps the REINFORCE form and clips only the weight: maximise sg(clip(ρ_{i,t}, 1 − ε_low, 1 + ε_high)) · Â_i · log π_θ(o_{i,t}), where sg is stop-gradient, so every token keeps a gradient.

### DPO and its gradient

$$
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\left(\hat{r}_w - \hat{r}_l\right),\quad \hat{r} = \beta\log\frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
$$

Here y_w and y_l are the chosen and rejected responses and r̂ is the implicit reward. Differentiate: with z = r̂_w − r̂_l, d(−log σ(z))/dz = −σ(−z), and ∇z = β[∇log π_θ(y_w) − ∇log π_θ(y_l)], so

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta\,\sigma(\hat{r}_l - \hat{r}_w)\left[\nabla_\theta\log\pi_\theta(y_w) - \nabla_\theta\log\pi_\theta(y_l)\right]
$$

The weight σ(r̂_l − r̂_w) is the probability the implicit reward model assigns to the wrong ordering. A pair at margin 0 gets 0.5; a pair already ranked right by 2 gets σ(−2) = 0.12; one ranked wrong by 2 gets 0.88. The bracket only asks for the gap to grow, so both log-probabilities can fall together (likelihood displacement, Ch 18).

### SimPO

$$
\mathcal{L}_{\mathrm{SimPO}} = -\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l \mid x) - \gamma\right)
$$

The reward is the length-normalised log-probability, the same quantity the model is scored on when it generates, so there is no reference model (half the memory) and no reward for length; γ > 0 is a target margin. ORPO instead adds −λ log σ(log odds(y_w) − log odds(y_l)) to the SFT loss, with odds(y) = P(y)/(1 − P(y)) and P the length-normalised likelihood.

### Other post-training formulas

- GAE: δ_t = r_t + γV(s_(t+1)) − V(s_t); A_t = δ_t + γλ A_(t+1). λ = 0 gives the one-step TD error (low variance, biased by V); λ = 1 gives the Monte Carlo return minus V (unbiased, high variance)
- PPO-clip: max E[min(ρA, clip(ρ, 1−ε, 1+ε)A)], ρ = π/π_old
- KL estimators (r = π_ref/π): k1 = −log r; k2 = (log r)²/2; k3 = r − 1 − log r
- KL-regularised optimum: π*(y|x) ∝ π_ref(y|x) exp(r(x, y)/β)
- IPO: (h − 1/(2β))², with h = log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l)), the log-ratio margin of chosen over rejected; the loss pulls h towards the fixed target 1/(2β) instead of pushing it to infinity
- Best-of-n KL from the base policy: at most log n − (n − 1)/n; the classic formula is an upper bound, and the true KL is smaller when likely responses repeat. n = 16 → ≤ 1.835 nats
- Unbiased pass@k from n samples with c correct: 1 − C(n−c, k)/C(n, k). With n = 10 and c = 3: pass@1 = 0.30, pass@5 = 1 − 21/252 = 0.92
- Self-consistency: majority vote over sampled final answers
- REINFORCE: ∇J = E[(R(y) − b) ∇log π_θ(y | x)]; a baseline b that does not depend on y leaves it unbiased because E[∇log π] = ∇Σ π = 0
- RLOO baseline: b_i = mean of the other G − 1 rewards for the same prompt
- KTO: weights each single response by whether it was labelled desirable or undesirable, using the implicit reward r̂ = β log(π_θ/π_ref) against a reference point estimated from the KL of the batch; no pairs needed

## Evaluation and interpretability (Ch 23, 26, 27)

- Clustered SE of a mean: √(Σ over clusters c of (Σ of e_i for i in c)²) / n, with e_i = x_i − x̄ the residual of item i; use it when items share a passage, a template or a source document
- Paired sample size: n ≈ (z_(α/2) + z_β)² · d / δ², with disagreement rate d
- Bits per byte: (Σ token loss in nats) / (bytes × ln 2)
- 2PL IRT: P(correct) = σ(a_i(θ_m − b_i)), with θ_m the ability of model m, b_i the difficulty of item i and a_i its discrimination (how sharply the item separates weaker from stronger models)
- Bradley–Terry / Elo: P(i beats j) = σ(s_i − s_j); Elo scale 400/ln 10 per unit
- SAE: f = ReLU(W_e(x − b_d) + b_e), x̂ = W_d f + b_d; loss |x − x̂|² + λ Σ f_i |W_d,i| (or TopK)
- Weak-to-strong PGR = (student − weak) / (ceiling − weak)
- Zero successes in n trials: 95% upper bound 1 − 0.05^(1/n) ≈ 3/n

## Inference (Ch 21)

- Temperature: p_i ∝ exp(z_i / T)
- KV cache bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_value
- Weights memory = parameters × bytes per parameter (fp16 2, int8 1, int4 0.5)
- Inference FLOPs per generated token ≈ 2N for N (active) parameters, plus about 4 · layers · T · d for attention over a context of T tokens: for an 8B model (32 layers, d = 4,096) at T = 8,192, the attention part is 4.3 GFLOPs against 16 GFLOPs for the weights, about a quarter extra
- Roofline: attainable FLOP/s = min(peak, I × bandwidth), with I the arithmetic intensity in FLOPs per byte moved. The ridge point peak/bandwidth is about 989e12 / 3.35e12 ≈ 295 for an H100 in bf16. Decode with bf16 weights at batch b has I ≈ b, so at b = 32 it reaches at most 32 × 3.35e12 ≈ 1.1e14 FLOP/s, 11% of peak
- Decode speed bound (single stream) ≈ memory bandwidth / bytes read per token
- Speculative decoding: expected tokens per target pass = (1 − α^(k+1)) / (1 − α), with α the per-token acceptance rate and k the draft length; each draft token x is accepted with probability min(1, p(x)/q(x)) and, on rejection, a token is drawn from norm(max(0, p − q)), which makes the output exactly the target distribution p (q is the draft's)
- Latency = TTFT + output_tokens × TPOT
- Cost = requests × (tokens_in × price_in + tokens_out × price_out) / 10⁶

## Retrieval and recommendation (Ch 29, 30, 43)

- Cosine similarity: (a · b) / (‖a‖ ‖b‖); equals the dot product for unit vectors
- TF-IDF: tf(t, d) × log(N / df(t))
- BM25: Σ_t IDF(t) × f(t, d)(k₁ + 1) / (f(t, d) + k₁(1 − b + b |d| / avgdl)), typically k₁ ≈ 1.2–2, b ≈ 0.75
- Reciprocal rank fusion: score(d) = Σ_r 1 / (k + rank_r(d)), k ≈ 60
- InfoNCE: L = −log [exp(sim(q, d⁺)/τ) / Σ_j exp(sim(q, d_j)/τ)]
- Matrix factorisation: r̂_ui = p_u · q_i + b_u + b_i + μ
- BPR loss: −log σ(r̂_ui − r̂_uj) for observed i, unobserved j
- Off-policy evaluation by inverse propensity scoring: V̂_IPS = (1/n) Σ y_i · π_new(a_i | x_i) / π_log(a_i | x_i), the value of a new policy from logs of an old one, where x_i is the context, a_i the logged action (item shown, slot), y_i its reward and π_log the logging policy's probability of that action. Unbiased only if π_log > 0 wherever π_new > 0, and the probabilities must be logged at decision time
- Clipped IPS: replace each weight w_i = π_new/π_log by min(w_i, M), trading a little bias for much less variance. Self-normalised IPS (SNIPS): Σ w_i y_i / Σ w_i. Example: three logs with (y, π_log, π_new) = (1, 0.5, 0.9), (0, 0.25, 0.05), (1, 0.25, 0.05) have weights 1.8, 0.2, 0.2; IPS = (1.8 + 0.2)/3 = 0.67, SNIPS = 2.0/2.2 = 0.91
- Position-bias IPS for learning to rank: weight each clicked item's loss by 1/P(examined at its position); it applies to losses over clicked items, so pointwise losses on non-clicks need a separate correction (Ch 43, 44)
- Thompson sampling (Bernoulli): sample θ_k ~ Beta(1 + successes_k, 1 + failures_k), pick argmax

---

[← B · Cheat sheets](B-cheat-sheets.md) · [Contents](../README.md#contents) · [D · Paper map →](D-paper-map.md)
