# 1 · The maths you actually use

Part I · Foundations. The maths behind the rest of the book, limited to what practitioners use. Every idea here shows up later: cosine similarity in retrieval, SVD in matrix factorisation and LoRA, Bayes in naive Bayes and calibration, cross-entropy and KL in every model you train.

## Why it matters

You do not need a maths degree to be a strong applied ML engineer, but you need fluent intuition for a small set of ideas, because interviewers use them to test whether you understand models or only call them. "Why cosine and not Euclidean?", "what is LoRA doing in linear-algebra terms?", "why is cross-entropy the classification loss?" all come from this chapter.

## Key concepts

- **Vector, dot product, cosine similarity**: a vector is a list of numbers (an embedding, a feature row); the dot product measures how aligned two vectors are; cosine similarity is the dot product of their length-normalised versions. [3Blue1Brown: Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra).
- **Matrix multiplication**: every neural-network layer is one; it maps vectors from one space to another. Same series as above.
- **Eigenvectors and SVD**: directions a matrix only stretches; SVD factorises any matrix into rotations and stretches and gives the best low-rank approximation. [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).
- **PCA**: SVD of centred data; finds the directions of largest variance. [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis).
- **Bayes' rule and base rates**: how to update a probability with evidence, and why rare events produce many false alarms. [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).
- **MLE and MAP**: choose parameters that make the data most likely; MAP adds a prior, which is where regularisation comes from. [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), [MAP estimation](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation).
- **Gradient and chain rule**: the direction of steepest increase, and how derivatives compose through layers; together they give gradient descent and backpropagation. [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).
- **Entropy, cross-entropy, KL divergence**: measures of uncertainty and of the gap between two distributions; cross-entropy is the standard classification loss. [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
- **Whole-subject reference**: [Mathematics for Machine Learning](https://mml-book.github.io/) (Deisenroth, Faisal and Ong), free online.

## 1.1 Vectors, dot products and cosine similarity

A vector is a list of numbers; in ML, a point in a feature or embedding space. The **dot product** a·b = Σaᵢbᵢ measures alignment weighted by length; **cosine similarity** a·b / (‖a‖‖b‖) measures only the angle: 1 for the same direction, 0 for orthogonal, −1 for opposite.

```python
import numpy as np

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

cosine(np.array([1., 2, 0]), np.array([2., 4, 0]))   # 1.0: same direction, different length
cosine(np.array([1., 0]), np.array([0., 1]))         # 0.0: orthogonal
```

**Why cosine for text embeddings:** vector length often reflects things like text length or frequency rather than meaning; cosine ignores it. On L2-normalised vectors, cosine, dot product and (monotonically) Euclidean distance give the same ranking, which is why vector databases normalise.

**Norms:** L2 (Euclidean length) and L1 (sum of absolute values) appear as regularisers (Chapter 3.4). In high dimensions, distances between random points concentrate (all look similar), one reason kNN and clustering struggle without good representations.

**Geometry in high dimensions.** Intuitions from 2-D and 3-D mislead in the 768- or 4,096-dimensional spaces of embeddings. Take two random vectors with independent standard-normal entries. Their dot product is a sum of d independent products, so its spread grows like √d, while their lengths grow like √d each; the cosine therefore shrinks like 1/√d. Measured on 20,000 random pairs, the standard deviation of the cosine is 0.578 in 3 dimensions, 0.125 in 64 and 0.036 in 768, matching 1/√d. In high dimensions, almost every pair of random directions is nearly orthogonal.

![Distribution of the cosine between two random vectors (computed in tools/figures.py)](images/01-fig-1.png)

*Distribution of the cosine between two random vectors (computed in `tools/figures.py`). In 768 dimensions nearly all pairs fall within ±0.1.*

Two consequences follow. A cosine of 0.3 between two embeddings is a strong signal, not a weak one, because chance alone almost never produces it. And trained embedding spaces are usually **anisotropic**: real embeddings crowd into a narrow cone, so even unrelated texts can score 0.6–0.8. Similarity thresholds must therefore be calibrated per model on labelled pairs, never carried over from another model (Chapter 29).

## 1.2 Matrices, eigenvectors and the SVD

**Matrix multiplication** is the core operation of every model: a linear layer computes Wx; a batch is XWᵀ; attention is QKᵀ. Track shapes: (n × d) @ (d × k) = (n × k). Most bugs in from-scratch code are shape bugs.

**Eigenvectors** of a square matrix are directions it only stretches: Av = λv. For a covariance matrix, eigenvectors are the principal components and eigenvalues the variance along them (PCA, Chapter 40.10).

**The singular value decomposition** factors any matrix: A = UΣVᵀ, with orthonormal U and V and non-negative singular values in Σ, largest first. Keeping the top k gives the **best rank-k approximation** (Eckart–Young theorem).

**What the SVD says geometrically.** Every matrix, applied to a vector, does three things in sequence: Vᵀ rotates the input into a special set of axes, Σ stretches each axis by its singular value, and U rotates the result into the output space. A circle of input vectors becomes an ellipse whose semi-axes are the singular values. For the 2 × 2 matrix [[3, 0], [4, 5]], the singular values are 6.708 and 2.236: the matrix stretches one direction by about 6.7 and another by about 2.2. Keeping only the first gives a rank-1 approximation with 31.6% relative error, because the second stretch is not negligible. Real data matrices typically have a few large singular values and a long tail of small ones, which is exactly when truncation works.

**Why real matrices are approximately low rank.** A user–item matrix is low rank when a few hidden factors (price sensitivity, cuisine preference, time of day) explain most behaviour; each user's row is then roughly a mix of a few typical rows. The same holds for the weight *change* when fine-tuning a large model on a narrow task: LoRA bets that the update has low effective rank. For a 4,096 × 4,096 weight matrix, a rank-16 update has 16 × (4,096 + 4,096) = 131,072 parameters instead of 16,777,216: 0.78% of them (Chapter 20).

![truncated SVD · a 100 × 80 matrix at rank 5](images/01-01.png)

*truncated SVD · a 100 × 80 matrix at rank 5*

```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
A_k = U[:, :k] * S[:k] @ Vt[:k]                 # rank-k reconstruction
rel_error = np.linalg.norm(A - A_k) / np.linalg.norm(A)
```

On a 100 × 80 test matrix built as rank 5 plus noise, rank 1 left 83% relative error, rank 5 only 4.6%, and rank 20 3.3% (the rest is noise). Storage at rank 5: 900 numbers instead of 8,000.

This single idea explains **matrix factorisation** for recommendations (Chapter 43.2: the user-item matrix is approximately low rank), **LoRA** (Chapter 20.3: the fine-tuning update is approximately low rank), **PCA**, and embedding compression.

## 1.3 Probability, Bayes and common distributions

**Bayes' rule:** P(A|B) = P(B|A)·P(A) / P(B). The practical lesson is the **base rate**: a fraud model with 95% recall and a 5% false positive rate, on a population with 1% fraud, gives P(fraud | flagged) = 0.95·0.01 / (0.95·0.01 + 0.05·0.99) ≈ 16%. Most flags are false alarms, which is why precision collapses for rare events (Chapter 4.2).

**The same calculation in natural frequencies**, which is how to explain it to anyone: of 10,000 orders, 100 are fraudulent and 9,900 are not. The model catches 95 of the 100 frauds, and wrongly flags 5% of the 9,900 good orders, which is 495. Of the 590 flagged orders, only 95 are fraud: 16%. Halving the false positive rate helps more than raising recall, because the false alarms come from the huge majority class.

![Precision of the same detector (95% recall) as prevalence changes](images/01-fig-2.png)

*Precision of the same detector (95% recall) as prevalence changes. At 0.1% prevalence it is 1.9% with a 5% false-positive rate and 8.7% with 1%; at 10% prevalence, 68% and 91%.*

| Distribution | Models | Used for |
|---|---|---|
| Bernoulli / binomial | Yes/no outcomes, counts of successes | Clicks, conversions; logistic regression's likelihood |
| Categorical / multinomial | One of k outcomes | Next-token prediction; softmax classifiers |
| Poisson | Counts of events in an interval | Orders per hour, demand forecasting |
| Normal (Gaussian) | Sums of many small effects | Measurement noise, the CLT, regression errors |
| Log-normal | Products of factors; skewed positive values | Basket values, delivery durations |
| Beta | A probability itself | Prior for click rates; Thompson sampling |
| Exponential | Waiting times | Time between orders |

**Expectation and variance** rules worth knowing: E[aX + b] = aE[X] + b; Var(aX) = a²Var(X); variances of independent variables add; the variance of a mean of n independent draws is σ²/n (why bigger samples give tighter estimates).

## 1.4 MLE and MAP

**Maximum likelihood (MLE):** choose parameters that make the observed data most probable. For a click rate with 3 clicks in 10 impressions, MLE = 0.3.

**The derivation**, because the same steps recur everywhere: with k clicks in n impressions and click probability θ, the likelihood is θᵏ(1 − θ)ⁿ⁻ᵏ. Take the log (products become sums and the maximum does not move): k log θ + (n − k) log(1 − θ). Set the derivative k/θ − (n − k)/(1 − θ) to zero and solve: θ = k/n. Training a classifier with log loss is this same calculation with θ produced by a model for each example instead of one shared number.

**Maximum a posteriori (MAP):** add a prior and maximise the posterior. With a Beta(2, 38) prior (mean 5%, a typical click rate), the MAP estimate is (3 + 2 − 1)/(10 + 2 + 38 − 2) ≈ 0.083, and the posterior mean (3 + 2)/(10 + 40) = 0.10. The prior pulls small-sample estimates towards the typical value: exactly the **smoothing** used for CTR features and target encoding (Chapters 3.8, 44.4).

**Why the Beta prior is convenient.** The Beta distribution is *conjugate* to the binomial: a Beta(a, b) prior plus k successes in n trials gives a Beta(a + k, b + n − k) posterior, so updating is just adding counts. The prior acts like a + b imaginary impressions with a imaginary clicks. With Beta(2, 38):

| Observed | Raw rate | Posterior mean (k + 2)/(n + 40) |
|---|---|---|
| 3 / 10 | 0.300 | 0.100 |
| 30 / 100 | 0.300 | 0.229 |
| 300 / 1,000 | 0.300 | 0.290 |
| 0 / 5 | 0.000 | 0.044 |

The same raw rate is trusted more as evidence accumulates, and an item with no clicks yet is not ranked as if it were certainly worthless. This is empirical Bayes: fit a and b to the distribution of rates across all items, then shrink each item towards it.

**Connections to remember:** minimising squared error = MLE under Gaussian noise; minimising log loss = MLE for Bernoulli labels; L2 regularisation = MAP with a Gaussian prior on weights; L1 = MAP with a Laplace prior.

## 1.5 Gradients and the chain rule

A gradient is the vector of partial derivatives: the direction of steepest increase. Training moves parameters against it (Chapter 3.5). The **chain rule** composes derivatives through nested functions, d/dx f(g(x)) = f′(g(x))·g′(x), and **backpropagation** applies it efficiently from the loss back through every layer (Chapter 5).

**The chain rule by hand.** Take one example with input x = 2, target y = 5 and the model z = w·x + b with w = 3 and b = 1, and the loss L = (z − y)². The forward pass gives z = 7 and L = 4. Going backwards: ∂L/∂z = 2(z − y) = 4; since ∂z/∂w = x and ∂z/∂b = 1, the chain rule gives ∂L/∂w = 4 × 2 = 8 and ∂L/∂b = 4. A gradient step with learning rate 0.01 moves w to 2.92 and b to 0.96, reducing z towards 5. Backpropagation is exactly this bookkeeping, done for millions of parameters by reusing the shared upstream factor (here ∂L/∂z) instead of recomputing it for each parameter.

The most useful derivative to know by heart: for logistic regression with log loss, the gradient is Xᵀ(σ(Xw) − y)/n, and the same "prediction minus target" form holds for softmax with cross-entropy.

**Check gradients numerically** when you implement them by hand: compare with central differences (f(w + ε) − f(w − ε))/2ε.

```python
def numerical_grad(f, w, eps=1e-6, *args):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = eps
        g[i] = (f(w + e, *args) - f(w - e, *args)) / (2 * eps)
    return g

np.allclose(analytic_grad(w, X, y), numerical_grad(log_loss, w, 1e-6, X, y), atol=1e-6)   # True
```

## 1.6 Entropy, cross-entropy and KL divergence

- **Entropy** H(p) = −Σ p log p: the average surprise of a distribution; zero when certain, maximal when uniform.
- **Cross-entropy** H(p, q) = −Σ p log q: the average surprise when reality follows p but you predict q. Classification loss is cross-entropy between the one-hot label and the predicted distribution.
- **KL divergence** KL(p ‖ q) = H(p, q) − H(p) ≥ 0: the extra cost of using q instead of p. Not symmetric.

```python
def entropy(p):          return -np.sum(p * np.log2(p))
def cross_entropy(p, q): return -np.sum(p * np.log2(q))
def kl(p, q):            return cross_entropy(p, q) - entropy(p)

p, q = np.array([0.7, 0.2, 0.1]), np.array([0.5, 0.3, 0.2])
# entropy(p) = 1.157 bits, cross_entropy(p, q) = 1.28, kl(p, q) = 0.123, kl(q, p) = 0.133: not symmetric
```

**What cross-entropy charges for a prediction.** For one example, the loss is −log of the probability given to the true class:

| Probability on the true class | 0.9 | 0.5 | 0.1 | 0.01 |
|---|---|---|---|---|
| Cross-entropy (nats) | 0.105 | 0.693 | 2.303 | 4.605 |

Being confidently wrong is expensive and grows without bound, which is why cross-entropy pushes models towards honest probabilities, and why a handful of mislabelled examples can dominate a training loss. Label smoothing (training towards 0.9 on the true class instead of 1.0) caps how confident the model is asked to be.

Where KL appears: the RLHF penalty that keeps a fine-tuned policy close to the reference model (Chapter 15), distillation from a teacher's distribution, variational autoencoders, and monitoring drift between feature distributions.

**Numerical stability:** compute softmax as exp(z − max z) / Σ, otherwise large logits overflow (softmax of [1000, 1001, 1002] works only with the shift and gives [0.09, 0.245, 0.665]); compute log-softmax directly rather than log(softmax).

## 1.7 Matrix calculus and curvature

Research code and papers use a few matrix-calculus facts constantly:

- **Gradient shapes match parameter shapes.** For y = Wx with loss L, ∂L/∂W = (∂L/∂y) xᵀ (an outer product) and ∂L/∂x = Wᵀ(∂L/∂y). Backpropagation through a linear layer is these two lines.
- **Jacobians** collect all first derivatives of a vector function: J_ij = ∂y_i/∂x_j. The chain rule multiplies Jacobians; backpropagation computes vector–Jacobian products without forming them.
- **Softmax Jacobian**: ∂s_i/∂z_j = s_i(δ_ij − s_j). Combined with cross-entropy it collapses to the famous gradient s − y.
- **The Hessian** (second derivatives) describes curvature. Its largest eigenvalue (the sharpness) limits the stable learning rate of gradient descent to about 2/λ_max; the ratio of largest to smallest eigenvalue (the condition number) explains why plain gradient descent zig-zags in narrow valleys and why adaptive and preconditioned optimisers help (Chapter 7).
- **Convexity**: linear and logistic regression have convex losses (one minimum); neural networks do not, yet in high dimensions most critical points are saddles rather than bad local minima, which is part of why gradient descent works.

## 1.8 Information theory for language models

**Forward versus reverse KL.** KL divergence is not symmetric, and which direction you minimise changes the answer. Fit a single Gaussian q to a two-mode target p (modes at −2 and +2, each with standard deviation 0.6):

| Objective | Fitted mean | Fitted sd | Behaviour |
|---|---|---|---|
| Forward KL(p ‖ q) | 0.00 | 2.09 | Mode-covering: spreads to cover both modes, puts mass where p has none |
| Reverse KL(q ‖ p) | 2.00 | 0.60 | Mode-seeking: picks one mode and fits it tightly |

Maximum-likelihood training (cross-entropy on data) minimises forward KL from the data to the model, so language models cover the data distribution, including its unlikely corners. The KL penalty in RLHF is a reverse KL from the policy to the reference, and on-policy distillation also uses reverse KL: both make the student or policy mode-seeking, which tends to produce sharper but less diverse outputs (Chapters 18 and 19).

**Perplexity and bits.** A language model's cross-entropy loss in nats per token converts to bits per token by dividing by ln 2, and to **perplexity** by exponentiating: loss 2.0 nats is 2.89 bits per token and perplexity 7.39, as if the model were choosing uniformly among about 7 tokens at each step. Because tokens differ in length between tokenizers, compare models in **bits per byte** (Chapter 23).

**Compression is prediction.** A model that assigns probability p to the next symbol can encode it in −log₂ p bits with arithmetic coding, so a language model's total log-loss on a text is the length of the shortest code it can produce for it. Better prediction is better compression; this is why pretraining loss is such a robust measure of model quality, and why "the model learns to compress the internet" is a precise statement rather than a metaphor.

**Mutual information** I(X; Y) = H(X) − H(X | Y) measures how much knowing Y reduces uncertainty about X. It underlies contrastive learning (InfoNCE is a lower bound on mutual information, Chapter 29) and some interpretability and data-selection methods.

## Pitfalls

- **Ignoring base rates** when reading a classifier's flags.
- **Comparing unnormalised dot products** when lengths carry no meaning.
- **Shape errors** in matrix code; print shapes at every step when debugging.
- **Naive softmax** without the max shift; log of zero in cross-entropy.
- **Treating KL as a distance:** it is not symmetric.
- **Estimating rates from tiny samples** without a prior or smoothing.

## Interview questions

**1. [B] Cosine similarity vs Euclidean distance for embeddings?** Cosine ignores length and measures direction; on normalised vectors both give the same ranking. Use cosine when length is not meaningful.

**2. [B] What is the SVD and its main use?** A = UΣVᵀ; truncating to the top k singular values gives the best rank-k approximation: PCA, matrix factorisation, compression, LoRA's intuition.

**3. [I] A test is 95% sensitive with a 5% false positive rate; prevalence is 1%. What share of positives are real?** About 16%: 0.95 × 0.01 / (0.95 × 0.01 + 0.05 × 0.99).

**4. [I] Relate L2 regularisation to Bayesian inference.** It is MAP estimation with a zero-mean Gaussian prior on the weights; L1 corresponds to a Laplace prior.

**5. [I] Why is cross-entropy preferred to squared error for classification?** It is the negative log-likelihood of the labels, penalises confident mistakes strongly, and with softmax gives gradients (p − y) that do not vanish when predictions are wrong and saturated.

**6. [I] What does KL divergence measure, and where does it appear in LLM training?** The extra expected log-loss from using q instead of p; as the penalty keeping an RLHF policy close to the reference model, and in distillation.

**7. [S] Why does LoRA work, in linear-algebra terms?** The weight change needed for a task has low effective rank, so it is well approximated by a product of two thin matrices, cutting trainable parameters from d·k to r(d + k).

**8. [S] How would you smooth click-through rates for restaurants with few impressions?** A Beta prior fitted to the overall distribution (empirical Bayes), using the posterior mean (clicks + α)/(impressions + α + β), which shrinks small samples towards the global rate.

## Practice

1. Implement cosine similarity, matrix-vector products and a softmax with the max shift in NumPy, without looking.
2. Reproduce the low-rank experiment: build a rank-5 matrix plus noise, plot reconstruction error against k.
3. Derive the gradient of logistic regression's log loss by hand, then check it numerically.
4. Fit a Beta prior to the CTRs of many items (method of moments) and compare raw and smoothed CTR rankings.
5. Compute entropy, cross-entropy and KL for a few distributions and verify KL ≥ 0 and its asymmetry.

## Read next

- *Mathematics for Machine Learning* (Deisenroth, Faisal and Ong, free online): chapters on linear algebra, matrix decompositions, probability and continuous optimisation.
- 3Blue1Brown, *Essence of Linear Algebra* and *Essence of Calculus* (video series), for intuition.
- *Dive into Deep Learning*, the preliminaries chapter.
- Gilbert Strang's linear algebra lectures (MIT OpenCourseWare), for SVD in depth.

**Free online:** [Mathematics for Machine Learning](https://mml-book.github.io/) · [3Blue1Brown: linear algebra](https://www.3blue1brown.com/topics/linear-algebra) · [3Blue1Brown: neural networks and backpropagation](https://www.3blue1brown.com/topics/neural-networks). Book titles above can be found by searching their title.

## Done when

You can explain cosine similarity, matrix shapes and the SVD's best-rank-k property. You can apply Bayes with base rates in your head. You can connect MLE, MAP and regularisation, and smooth rates with a Beta prior.

You can derive and numerically check a simple gradient. You can define entropy, cross-entropy and KL and say where each appears in ML.

---

[← Overview](00-overview.md) · [Contents](../README.md#contents) · [2 · Statistics and experimentation →](02-statistics-and-experimentation.md)
