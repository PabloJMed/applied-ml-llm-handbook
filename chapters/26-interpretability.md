# 26 · Interpretability and analysis

Part VIII · Interpretability, alignment and safety. Assumes Chapter 10 (attention, the residual stream) and Chapter 6 (PyTorch hooks). Chapter 27 uses these tools for safety: detecting deception, monitoring reasoning, steering behaviour.

## Why it matters

We train models whose behaviour we can measure but whose internal computations we mostly cannot read. Interpretability tries to change that: find the features a model represents, the circuits that compute with them, and causal handles to change behaviour. It matters for science (what did the model actually learn?), for debugging (why does it fail on this input?), and for safety (can we tell when a model is being deceptive, or verify that a dangerous capability is absent?). Since 2023, sparse autoencoders and circuit-tracing methods have made it possible to find millions of interpretable features in production-scale models.

Research-scientist interviews increasingly include interpretability: explain the residual stream view, write down the QK and OV circuits, say what an induction head is, how activation patching works, and what sparse autoencoders do and fail to do. The interviewer is usually testing two things at once. The first is whether you can do the linear algebra: decompose a head into what it reads and what it writes, or derive why an L1 penalty shrinks features. The second is whether you have the scientific habits of the field: knowing which evidence is correlational and which is causal, and which famous results come with caveats. Applied and responsible-AI interviews at big tech ask about gradient attributions (saliency, integrated gradients, SHAP) more often than about SAEs, so this chapter covers both.

Numbers in this chapter come from `code/ch26_check.py` (CPU, a few minutes): a tiny transformer trained from scratch, analysed with the same methods used on large models.

The chapter starts with the questions interpretability asks and how strong each kind of evidence is (26.1), then introduces the frame most methods share: the residual stream as a running sum that every layer reads from and writes to, with attention heads split into QK and OV circuits (26.2). The methods come next, in order of increasing strength of evidence: probing what information is present (26.3), reading intermediate predictions with the logit lens (26.4), and causal interventions that test whether a component matters (26.5). Sections 26.6 and 26.7 show what these methods have found, from induction-head circuits to features recovered by sparse autoencoders, and 26.8 uses the results to steer models. Then come reasoning traces as evidence (26.9), good practice (26.10) and a worked investigation (26.11). The last two sections cover gradient-based attribution (26.12) and the tools for avoiding interpretability illusions and comparing models (26.13).

The central distinction is between correlation and causation. A probe that finds information in a layer shows the information is there, not that the model uses it. Only interventions, changing an activation and observing the behaviour change, show what a component does. Most disagreements in the field are about how much an observational result proves.

## Key concepts

- **Residual stream view**: each layer reads from and writes to a shared vector space. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
- **QK and OV circuits**: each attention head factors into W_QK = W_Qᵀ W_K (where to look) and W_OV = W_O W_V (what to move).
- **Induction heads**: a two-head circuit that implements in-context copying. [Olsson et al.](https://arxiv.org/abs/2209.11895).
- **Probing**: training simple classifiers on internal activations. [Control tasks (Hewitt and Liang)](https://arxiv.org/abs/1909.03368).
- **Logit lens and tuned lens**: decoding intermediate layers into vocabulary predictions. [Tuned lens](https://arxiv.org/abs/2303.08112).
- **Activation patching and causal interventions**. [ROME (causal tracing)](https://arxiv.org/abs/2202.05262), [IOI circuit](https://arxiv.org/abs/2211.00593), [Best practices for activation patching](https://arxiv.org/abs/2309.16042).
- **Superposition and sparse autoencoders (SAEs)**. [Toy Models of Superposition](https://arxiv.org/abs/2209.10652), [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), [Scaling and evaluating sparse autoencoders (TopK)](https://arxiv.org/abs/2406.04093).
- **Circuit tracing with transcoders and attribution graphs**. [Circuit tracing (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).
- **Steering and representation engineering**. [Activation addition](https://arxiv.org/abs/2308.10248), [Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717).
- **Gradient-based attribution**: saliency, [integrated gradients](https://arxiv.org/abs/1703.01365), SHAP, and their [sanity checks](https://arxiv.org/abs/1810.03292).
- **Tools**: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [nnsight](https://nnsight.net/), [SAELens](https://github.com/jbloomAus/SAELens), [Neuronpedia](https://www.neuronpedia.org/).

## 26.1 Questions and levels

"What is the model doing?" can be asked at many levels: what information a layer contains, which components a behaviour depends on, or the full algorithm a circuit implements. The methods for each level differ, and so does the strength of the conclusions they support.

| Question | Methods | Strength of evidence |
|---|---|---|
| Is information X present at layer l? | Probing | Correlational: present, not necessarily used |
| What does the model "think" at layer l? | Logit lens, tuned lens | Descriptive |
| Which input features drive this output? | Saliency, integrated gradients, SHAP | Descriptive, local, often fragile |
| Which components cause behaviour B? | Ablation, activation patching, attribution | Causal, at the component level |
| What algorithm implements B? | Circuit analysis, causal abstraction | Causal and mechanistic, usually narrow |
| What features does the model use? | Sparse autoencoders, transcoders | Descriptive, with causal checks via steering |
| Can we change B by editing internals? | Steering vectors, model editing | Causal, practical |

Good interpretability work moves from description to causal tests: a probe that finds a feature is a hypothesis; an intervention that changes behaviour in the predicted way is evidence.

## 26.2 The residual stream view

Almost every method in this chapter assumes one picture of a transformer, and interviewers ask for it directly: "explain the residual stream view". The picture is useful because it turns a deep network into a sum, and sums can be split into parts that are studied one at a time.

### The stream as a running sum

A transformer's state at each position is a vector, the **residual stream**. Each attention head and MLP reads from it (through a linear map, after a layer norm), computes, and adds its output back. Write x_l for the stream at one position entering layer l, and head_{l,h} for head h of that layer. A pre-norm layer does two additions:

$$
x_l^{\mathrm{mid}} = x_l + \sum_{h} \mathrm{head}_{l,h}(x_l)
$$

$$
x_{l+1} = x_l^{\mathrm{mid}} + \mathrm{MLP}_l(x_l^{\mathrm{mid}})
$$

Nothing is ever overwritten; each component only adds a vector. Unrolling the additions over all L layers, the final stream is the token and position embedding x₀ plus the output of every component:

$$
x_L = x_0 + \sum_{l=0}^{L-1} \left( \sum_{h} \mathrm{head}_{l,h} + \mathrm{MLP}_l \right)
$$

and the logits are a readout of that sum, logits = W_U · LN(x_L), where W_U is the unembedding matrix and LN the final layer norm. Because the layer norm divides by one scale per position, and that scale is roughly constant across the small changes being studied, the logits are approximately a sum of each component's own contribution W_U · c / σ. That is **direct logit attribution**: how much each head or MLP pushes each output token, read straight from its output. And because each component's output stays in the stream, any later component can read it: the stream is a shared communication channel, with a width (the model dimension d) far smaller than the total output of all components, so components write to and read from subspaces of it.

### QK and OV circuits

An attention head does two separable jobs: it decides where to look, and it decides what to move from there. Write x_i for the stream at the query position i and x_j at a key position j, d_h for the head dimension, and W_Q, W_K, W_V (d_h × d) and W_O (d × d_h) for the head's weights. The attention score is a bilinear form in the two positions' streams:

$$
s_{ij} = \dfrac{x_i^\top W_{QK}\, x_j}{\sqrt{d_h}}, \quad W_{QK} = W_Q^\top W_K
$$

The softmax over j turns the scores into weights A_ij, and the head writes the weighted sum of the attended positions, each transformed by one matrix:

$$
\mathrm{head}(x)_i = \sum_{j \leq i} A_{ij}\, W_{OV}\, x_j, \quad W_{OV} = W_O W_V
$$

So a head is fully described by two d × d matrices of rank at most d_h. **W_QK**, the QK circuit, reads the query and key positions and decides *where to look*. **W_OV**, the OV circuit, reads the attended position and decides *what to write*. The separate W_Q and W_K are not meaningful on their own (any invertible matrix can be moved between them); only the products are. For a one-layer attention-only model you can push the embedding W_E and unembedding W_U through these matrices: W_Eᵀ W_QK W_E is a vocabulary × vocabulary table of which token attends to which, and W_U W_OV W_E is a table of how attending to token t changes the logit of every output token. A copying head shows a large positive diagonal in that second table.

### Composition

In a deeper model, a later head's input includes the outputs of earlier heads, so its QK and OV circuits can read them in three places. **Q-composition**: the query reads an earlier head's output. **K-composition**: the key does. **V-composition**: the value does, so the head moves information an earlier head put there. Multi-step algorithms are built from such compositions. For a pair of heads, Elhage et al. measure K-composition as ‖W_QK² W_OV¹‖_F / (‖W_QK²‖_F ‖W_OV¹‖_F): how much of the later head's key reading lies in the subspace the earlier head writes, compared with what random matrices give.

**A worked example you can check by hand.** Take three tokens A, B, C and a 6-dimensional stream: dimensions 0–2 hold the current token (one-hot), dimensions 3–5 hold "the previous token". The sequence is A B C A.

1. A layer-0 **previous-token head** attends from each position to the one before it. Its W_OV maps dimensions 0–2 to 3–5, so it writes "previous token = A" into position 2 (B's position), "previous = B" into position 3 and "previous = C" into position 4.
2. A layer-1 **induction head** has W_QK with 5s linking the query's current-token dimensions to the key's previous-token dimensions. At the last position (current token A), the score with each key is 5 × [current token of the query equals previous token of the key]. Only position 2 (B, whose previous token is A) matches, so the scores are [0, 5, 0, 0].
3. The softmax gives position 2 a weight of e⁵/(e⁵ + 3) = 0.980. The head's W_OV copies the current token, so it writes 0.98 × B into the stream, and the model predicts B.

This is K-composition: the induction head's key reads what the previous-token head wrote, and the product W_QK¹ W_OV⁰ is non-zero. It implements [A][B] … [A] → [B], which 26.6 finds in a trained model. A caution from the check script: in its trained two-layer model, the weight-based K-composition scores barely separate the heads (about 0.13 for every pair, against 0.125 for random matrices), because the Frobenius norm averages over all 64 directions while the circuit uses a small subspace, and the model has no layer norm to keep unused directions small. Weight-based scores are a hint; the ablations and patching in 26.6 are what establish the circuit.

## 26.3 Probing

The simplest interpretability question is whether some piece of information is present in a model's internal state at all. A probe answers it by training a small classifier to read the information from the activations. The method is cheap and widely used, and its main danger is over-interpreting a positive result.

Train a linear classifier on a layer's activations to predict a property (part of speech, truth value, a board-game state). High accuracy means the information is linearly decodable there.

Two cautions, both visible in the check script, where the target is whether x₀·x₁ > 0, a non-linear function of the input:

| Representation | Probe accuracy |
|---|---|
| Raw input | 0.486 (not linearly decodable) |
| Hidden layer of a random network | 0.624 |
| Hidden layer of a network trained on the task | 1.000 |
| Same layer, control task with random labels | ≈ 0.5 (chance; 0.47–0.52 across runs) |

- **Even random features help**: a random non-linear layer makes many functions partly decodable. Compare against random-network and input baselines.
- **Control tasks**: a probe powerful enough to memorise random labels says little. Use simple (linear) probes and report selectivity: accuracy on the real task minus accuracy on a control.

And the central limitation: decodable is not the same as used. A model may represent a property it never uses for its output. Follow probes with interventions (26.5, 26.8).

## 26.4 Logit lens and tuned lens

The final layer of a language model turns the residual stream into next-token probabilities. Applying that same readout to earlier layers shows what the model would predict if it stopped there, which reveals how a prediction forms across depth.

The **logit lens** applies the final layer norm LN_f and the unembedding W_U to the stream x_l after layer l:

$$
\mathrm{logits}_l = W_U\, \mathrm{LN}_f(x_l)
$$

It often shows the answer forming gradually across layers: early layers predict generic continuations; the correct token rises in the middle-to-late layers. It works poorly on early layers of some models because their representations are not yet in the output basis, so the lens reads noise or the input token. The **tuned lens** fixes this by learning a small affine map (A_l, b_l) per layer, trained so that the lens's distribution at layer l matches the model's final distribution:

$$
\min_{A_l, b_l}\; \mathbb{E}_x\, \mathrm{KL}\left( p_L \,\Vert\, \mathrm{softmax}(W_U\, \mathrm{LN}_f(A_l x_l + b_l)) \right)
$$

where p_L is the model's actual output distribution. The affine map absorbs the change of basis between layers, so the tuned lens gives sensible predictions from early layers and its predictions improve smoothly with depth. Both are cheap, descriptive tools for asking "when does the model know?".

**A worked trace.** In the check script's two-layer induction model, the logit lens on tokens that can only be predicted by induction gives the whole story in three numbers: reading the stream straight after the embeddings gives 0.033 and after layer 0 gives 0.035, both at chance (0.031), because the answer has not been fetched yet; after layer 1, where the induction heads write, it jumps to 0.972, the model's full accuracy. In a large model the same trace on a factual prompt ("The Eiffel Tower is in the city of") typically shows generic tokens in early layers, the answer appearing in the top few in middle layers, and its probability climbing in the last layers.

## 26.5 Causal interventions

Observational methods show what information is present; they cannot show what the model uses. Interventions can: change or remove a component's contribution and measure what happens to the behaviour. They are the strongest evidence interpretability has, and the basis of circuit analysis.

**Ablation**: remove a component's contribution and measure the change. How you remove it matters: zero-ablation produces activations the model never sees; **mean-ablation** (replace with the average over a dataset) or **resample-ablation** (replace with the activation from a different input) are closer to the data distribution.

**Activation patching** compares a clean input (where the behaviour occurs) with a corrupted input (minimally changed so it does not). Run the corrupted input, but replace one component's activation with its value from the clean run. If the behaviour comes back, that component carries the information that matters.

### Metrics, directions and paths

**The metric.** Patching needs a continuous measure of the behaviour. The standard one is the **logit difference**, logit(correct) − logit(incorrect) for a pair of answers that the clean and corrupted inputs favour (in the IOI task, the indirect object's name minus the subject's name). It is more sensitive than accuracy, which only changes when the top token flips, and less affected by unrelated tokens than the probability. Results are reported as a **normalised patching effect**, which puts the clean run at 1 and the corrupted run at 0:

$$
\mathrm{NPE} = \dfrac{m_{\mathrm{patched}} - m_{\mathrm{corrupted}}}{m_{\mathrm{clean}} - m_{\mathrm{corrupted}}}
$$

From the table in 26.6: patching both induction heads gives (0.974 − 0.030)/(0.975 − 0.030) = 0.999, so the two heads together restore the behaviour fully. Patching L1H0 alone gives 0.731 and L1H1 alone 0.619. The single effects sum to 1.35, more than 1: either head alone recovers most of the behaviour, so their effects overlap rather than add. That is the redundancy the ablation table also shows.

**Denoising versus noising.** Patching a clean activation into a corrupted run (**denoising**, the direction above) asks whether the component is *sufficient*, given everything else corrupted, to restore the behaviour. Patching a corrupted activation into a clean run (**noising**) asks whether it is *necessary*: does losing it break the behaviour? The two can disagree. With redundant components, denoising one restores a lot while noising it breaks little, because the other still works; with components that only work together (an AND), denoising one restores nothing while noising it breaks everything. Report both when the conclusion depends on it. Ablation is a form of noising.

**Path patching** patches a component's output only along one path, for example only into the keys of one later head, leaving its effect on everything else unchanged. It tests a specific edge of a circuit ("the previous-token head matters *through* the induction head's keys"), which ordinary patching cannot separate from its other effects. The IOI circuit in GPT-2 small was mapped this way.

### Attribution patching

Patching every component one by one costs one forward pass per component, which is too slow for a large model. **Attribution patching** replaces each patch by a first-order Taylor estimate. Write a for a component's activation, m for the metric, and take the gradient of m with respect to a in the corrupted run; then

$$
m(a_{\mathrm{clean}}) - m(a_{\mathrm{corr}}) \approx (a_{\mathrm{clean}} - a_{\mathrm{corr}})^\top\, \nabla_a m
$$

with the gradient evaluated at the corrupted run. One forward pass on each input and one backward pass give the estimate for every component at once. The approximation is good when the patch is a small change in a region where m is close to linear, and it fails where the model is strongly non-linear between the two activations: attention patterns that saturate (a softmax that moves from attending mostly to one token to mostly another), layer norms whose scale changes, and components whose effect is an all-or-nothing switch. In the check script, with the mean log-probability of the right token as the metric (clean −0.34, corrupted −3.88), real patching of L1H0 and L1H1 raises it by 2.37 and 2.07, while attribution patching estimates 3.44 and 2.85. It ranks the two heads correctly but overstates both effects by about 40%, because restoring a head moves its attention from diffuse to sharp, far outside the region where the linear approximation holds. Use it to shortlist components, then confirm with real patching.

**Pitfalls**: results depend on the corruption (a different corruption isolates a different mechanism), on the metric (logit difference is more sensitive than accuracy), and on backup mechanisms (ablate one head and another takes over). Report several corruptions and metrics.

## 26.6 Circuits: induction heads

The check script trains a two-layer, attention-only transformer (two heads per layer) on sequences in which a random segment of random length is repeated: [x₁ … x_L … x₁ … x_L]. The repeated tokens can only be predicted by finding the earlier occurrence of the current token and copying what came after it. The trained model gets 97.5% of them right (chance: 3.1%).

Attention patterns reveal the mechanism:

| Head | Attends to the previous token | Attends to "the token after the previous occurrence" |
|---|---|---|
| L0H0 | 0.50 | 0.00 |
| L0H1 | 0.05 | 0.05 |
| L1H0 | 0.06 | 0.77 |
| L1H1 | 0.06 | 0.75 |

This is the **induction circuit** (Olsson et al., 2022). A **previous-token head** in layer 0 (L0H0) writes "the token before me was A" into each position. **Induction heads** in layer 1 use that through K-composition: at a position holding A, they look for positions whose previous token was A, which is the position right after the earlier A, and copy its token (B) to the output. That implements [A][B] … [A] → [B]. In QK and OV terms (26.2): the induction head's QK circuit matches "my current token" against "your previous token", and its OV circuit is a copying circuit, writing the attended token's identity towards the output.

Ablations and patching confirm it causally:

| Intervention | Accuracy on repeated tokens |
|---|---|
| None | 0.975 |
| Ablate L0H0 (previous-token head) | 0.245 |
| Ablate L0H1 | 0.926 |
| Ablate L1H0 / L1H1 (one induction head) | 0.708 / 0.874 |
| Ablate both induction heads | 0.037 |
| Corrupt the first occurrence | 0.030 |
| … and patch in clean L1H0 / L1H1 / both | 0.721 / 0.615 / 0.974 |

The two induction heads are partly redundant (removing one hurts, removing both destroys the behaviour), a common pattern that single-component ablations can miss. In large models, induction heads form abruptly early in training, at the same time as a jump in in-context learning ability: a visible bump in the training loss, which Olsson et al. call a phase change. Other well-studied circuits include indirect object identification in GPT-2 small (IOI) and factual recall, where causal tracing located facts in mid-layer MLPs (ROME).

## 26.7 Superposition and sparse autoencoders

A natural hope is that each neuron represents one concept. In practice most neurons respond to several unrelated things, because models represent more concepts than they have neurons. Sparse autoencoders are the main tool for recovering those concepts as directions in activation space, and they made large-scale feature discovery possible.

Individual neurons are often **polysemantic**: one neuron responds to unrelated concepts. The leading explanation is **superposition**: a model represents more features than it has dimensions, as nearly-orthogonal directions, which works when features are sparse (rarely active together). Toy Models of Superposition showed this happens in small networks and depends on sparsity and importance.

![Absolute cosine between random unit directions](images/26-fig-1.png)

*Absolute cosine between random unit directions. The typical pair (√(2/πd)) is close to orthogonal already at a few hundred dimensions. Even the worst-aligned pair among 10,000 directions has cosine below 0.09 at 4,096 dimensions, the width of a mid-sized model's residual stream.*

This geometry is what makes superposition possible. A residual stream of 4,096 dimensions can hold far more than 4,096 directions with little interference, and the number grows exponentially with the width if a fixed amount of interference is acceptable. The interference shows up as noise when several features are active at once, so superposition works when features are sparse. The model's nonlinearities then filter the small interference terms.

### Why the count grows exponentially

The "exponentially many" claim follows from a short Johnson–Lindenstrauss-style argument. For two random unit vectors in d dimensions, the cosine is approximately normal with mean 0 and variance 1/d, so the chance that it exceeds ε in absolute value is at most about 2 exp(−dε²/2). Among N random directions there are about N²/2 pairs. If N² exp(−dε²/2) < 1, the expected number of pairs with |cos| > ε is below one, so some set of N directions has every pair within ε of orthogonal. Solving for N:

$$
N \approx \exp(d\,\varepsilon^2 / 4)
$$

For d = 4,096 and ε = 0.1 the bound gives about 28,000 directions; for ε = 0.2 it gives about 6 × 10¹⁷. The bound is loose (the figure shows 10,000 random directions already staying below 0.09), but the exponential dependence on d is the point: allowing a little interference buys an enormous number of almost-independent directions.

### When superposition happens

The toy models of Elhage et al. (2022) make the conditions precise. A small ReLU network is trained to reconstruct n features through m < n hidden dimensions, where each feature has an **importance** (its weight in the loss) and a **sparsity** (the probability it is zero). The trained models fall into three regimes, which the paper draws as a phase diagram over sparsity and importance. When features are dense, the model gives the m most important features their own dimensions and *drops* the rest, which is what PCA would do. As features become sparser, less important features are *stored in superposition*, first as antipodal pairs sharing one dimension and then in more crowded geometric arrangements, because two sparse features are rarely active together and their interference rarely costs anything. Features whose importance is too low are still dropped. The nonlinearity is essential: with a purely linear model, interference is never filtered and superposition never pays, so the model behaves like PCA. Polysemantic neurons follow directly. If features are directions that are not aligned with the neuron basis, each neuron's activation is a projection onto many feature directions, so it fires for several unrelated features.

### The sparse autoencoder

If features are directions in superposition, we need to recover those directions from activations. A **sparse autoencoder (SAE)** learns an over-complete dictionary: encode an activation into many latents with a sparsity constraint, decode back as a sparse combination of learned directions. For an activation x of dimension d, with M ≫ d latents, the standard (L1) SAE is

$$
f = \mathrm{ReLU}(W_{\mathrm{enc}}(x - b_{\mathrm{dec}}) + b_{\mathrm{enc}})
$$

$$
\hat{x} = W_{\mathrm{dec}}\, f + b_{\mathrm{dec}}
$$

where f is the vector of M latent activations, W_enc (M × d) and b_enc are the encoder, and the columns W_dec,i of the decoder (d × M) are the learned feature directions. It is trained on the loss

$$
\mathcal{L} = \Vert x - \hat{x} \Vert_2^2 + \lambda \sum_{i} f_i\, \Vert W_{\mathrm{dec},i} \Vert_2
$$

The first term asks for faithful reconstruction; the second penalises the total activation of the latents, weighted by their decoder norms so the model cannot dodge the penalty by shrinking f and growing the decoder. λ sets the trade-off between sparsity and reconstruction.

**Why L1 shrinks the features.** Take one active latent with a unit-norm decoder column, and suppose the input's component along that direction is a. The latent's best value minimises (a − f)² + λf over f ≥ 0. Setting the derivative −2(a − f) + λ to zero gives

$$
f^{*} = \max(0,\; a - \lambda/2)
$$

So every active latent is pulled down by λ/2, whatever its true size: with a = 1 and λ = 0.3, the SAE reports 0.85. This is the soft-thresholding operator of the lasso, and it biases every reconstruction towards zero. Small true activations below λ/2 are zeroed entirely. In the check script, on inputs with a single active feature, the projection of the reconstruction onto the input, ⟨x̂, x⟩/⟨x, x⟩, is 0.65 for the λ = 0.3 SAE and 0.95 for the TopK SAE: the L1 reconstructions are a third too short.

**TopK and JumpReLU avoid it.** A **TopK SAE** (Gao et al., 2024) drops the penalty and keeps only the k largest pre-activations, f = TopK(ReLU(W_enc(x − b_dec) + b_enc)). Sparsity is fixed at exactly k, and the latents that survive are not penalised for their size, so there is no shrinkage. A **JumpReLU SAE** keeps a latent only if its pre-activation z exceeds a learned threshold θ, f = z · H(z − θ) with H the step function, and penalises the number of active latents (an L0 penalty) directly, training the thresholds with straight-through gradient estimates. Above the threshold the value passes unshrunk. Google DeepMind's Gemma Scope SAEs use JumpReLU. **Gated SAEs** separate the decision "is this latent active?" from the estimate of its size, with the L1 penalty applied only to the gate, for the same reason.

**Dead latents.** Some latents stop firing on any input early in training and never recover, because a latent that never fires gets no reconstruction gradient. In large SAEs a large share of latents can die. The remedies are to **resample** dead latents periodically (re-initialise their encoder and decoder towards inputs the SAE currently reconstructs badly), or to add an auxiliary loss, as Gao et al.'s AuxK does, that asks the dead latents to model the reconstruction error.

### Results on the toy and at scale

In the check script, 24 sparse features (each active 5% of the time, 1.2 on average per input) are stored in an 8-dimensional space (mean |cos| between feature directions 0.29: heavily overlapping). Three SAEs with 32 latents:

| SAE | Active latents per input (L0) | Unexplained variance | Features recovered (cos > 0.9) |
|---|---|---|---|
| L1 penalty 0.01 | 8.08 | 0.001 | 3 / 24 |
| L1 penalty 0.3 | 2.54 | 0.139 | 22 / 24 |
| TopK, k = 2 | 2.00 | 0.100 | 23 / 24 |

The first reconstructs perfectly and finds almost nothing: with weak sparsity, a dense code is easiest. Good reconstruction is not the goal; **sparse, faithful** reconstruction is. TopK SAEs (keep the k largest latents, no L1 penalty) make the sparsity level a direct choice and avoid shrinking active latents, and are widely used. Note that the TopK SAE reaches lower unexplained variance than the λ = 0.3 SAE with fewer active latents, which is the shrinkage cost made visible.

At scale, SAEs on a production model's residual stream found millions of features, many of them interpretable and multilingual and multimodal (a feature for a landmark firing on text and images of it), with causal effects when amplified. **Transcoders** replace an MLP with a sparse, wider MLP trained to imitate it, so features describe computation rather than just representation; **attribution graphs** built from them (2025) trace, for a specific prompt, which features influence which, producing readable pictures of multi-step computations such as planning a rhyme or doing two-hop reasoning.

**Open problems and evaluation.** SAEs miss features and invent others (feature splitting and absorption), the reconstruction error contains unexplained computation, and interpretability of features is judged by humans or LLMs with their own biases. Evaluate SAEs on: downstream loss when the SAE reconstruction replaces the real activation, sparsity, interpretability scores, and usefulness on concrete tasks (probing, steering, finding a known feature). Several 2025 studies found that simple baselines (linear probes) match SAE-based methods on some practical tasks: treat SAEs as one tool, not the answer.

## 26.8 Steering and editing

If a concept corresponds to a direction in the residual stream, then adding or removing that direction should change the model's behaviour. Steering tests that prediction and, when it works, offers a way to control behaviour without retraining.

### The linear representation hypothesis

The assumption behind steering has a name: the **linear representation hypothesis**, that high-level concepts are represented as directions (or low-dimensional subspaces) of activation space, so that a concept's presence is read by a projection and changed by adding a vector. The evidence for it is broad: linear probes recover many concepts, word-embedding analogies work by vector arithmetic, SAEs find directions that behave like concepts, and steering by adding a direction often works. It is not universal. Some features are multi-dimensional: days of the week and months of the year have been found arranged on circles in some models, and arithmetic on them uses that geometry. Treat "the concept is a direction" as a hypothesis that each steering or probing result tests, not as a fact.

### Steering vectors

The simplest steering vector is a difference of means. Collect activations at one layer for two contrasting sets of prompts, such as "love" versus "hate", or honest versus dishonest completions, and take v = mean(activations on set A) − mean(activations on set B). At inference, add a multiple of it at that layer, at every position:

$$
x_l \leftarrow x_l + \alpha\, v
$$

The method is effective, cheap and often surprisingly precise, but the coefficient α needs a sweep. Too small and nothing changes; too large and the added vector dominates the stream, so fluency degrades, the model repeats the concept regardless of context, and eventually produces nonsense. A useful protocol sweeps α over a range such as 0 to 2 times the typical norm of the stream and plots two curves: the size of the intended effect (the share of outputs that show the concept) and a cost (the increase in loss on unrelated text). The usable range lies where the first is high and the second small, and its width is itself a finding. The choice of layer matters as much: early layers carry token-level features, late layers are close to the output, and middle layers usually steer abstract behaviour best.

### Ablating a direction: refusal

In many chat models, refusal of harmful requests is mediated by roughly one direction (Arditi et al., 2024). With r̂ the unit vector along the difference of means between harmful and harmless prompts, **directional ablation** removes its component from the stream at every layer and position:

$$
x \leftarrow x - \hat{r}\,\hat{r}^\top x
$$

Removing it disables refusals; adding it causes refusal of harmless requests. Because the projection can be folded into the weights, the edit is a permanent change to an open-weight model's parameters. The result is a striking demonstration of both the power of linear interventions and the fragility of safety training in open-weight models, and a reason refusal training alone is not a safeguard for released weights (Chapter 27).

### SAE features and model editing

**SAE feature steering** clamps an SAE feature to a high value, which adds its decoder direction to the stream. In the check script, adding feature 3's direction to a zero activation makes the TopK SAE's top latent the one aligned with feature 3 (cosine 1.00), the toy version of finding and turning up a concept. Its failure mode is the same as for steering vectors, plus one of its own: a feature that looked monosemantic in its top activations may have other effects when clamped, so check the effect on unrelated text.

**Model editing** (ROME, MEMIT) applies rank-one updates to MLP weights to change specific facts, treating a mid-layer MLP as a key-value store in which the subject's representation is the key. Edits generalise imperfectly: an edited fact is often not used in multi-hop reasoning or in paraphrases, and edits can have side effects on related facts. Evaluate an edit on paraphrases, on related facts that should not change, and on downstream reasoning that should change.

## 26.9 Reasoning traces as an interpretability signal

Reasoning models produce long chains of thought in natural language. That is an opportunity: monitoring the chain of thought can reveal reward hacking or intent to deceive. It has a limit: the chain of thought is not guaranteed to be **faithful** to the computation; models can reach answers for reasons they do not state, and training pressure on the chain of thought can teach models to hide intent. Chapter 27 covers chain-of-thought monitoring as a safety tool.

### Measuring faithfulness

Faithfulness cannot be read off the text, so the methods intervene on the reasoning or on the input and check whether the stated reasoning tracks the change.

**Intervening on the chain of thought.** Lanham et al. (2023) truncated chains of thought partway, added mistakes to them, or replaced them with filler text, and measured how often the final answer changed. If the answer does not change when the reasoning is cut short or corrupted, the reasoning was not what produced it. They found the dependence varied widely by task, and that on some tasks larger models relied less on their stated reasoning than smaller ones.

**Biasing the input.** Turpin et al. (2023) added biasing features to few-shot prompts, for example reordering the examples so the correct answer was always "(A)", or stating that a user thought the answer was a particular option. Models shifted their answers towards the bias, accuracy fell by up to about 36% on some tasks, and the chains of thought rationalised the biased answer without mentioning the bias.

**Hint disclosure in reasoning models.** A 2025 study by Anthropic (Chen et al., "Reasoning models don't always say what they think") inserted hints into prompts, such as a note giving the answer or metadata containing it, and kept the cases where the model's answer changed to match the hint. It then measured how often the chain of thought mentioned the hint. The reported disclosure rates were low: about 25% for Claude 3.7 Sonnet and 39% for DeepSeek R1, averaged over hint types. Outcome-based RL improved disclosure at first and then plateaued.

The conclusion for practice: a chain of thought is evidence about the computation, often useful, sometimes misleading, and never proof. Monitoring it catches a lot (Chapter 27.8), but the absence of a stated intention is weak evidence that the intention is absent.

## 26.10 Good practice

Interpretability results are easy to over-claim, and the field has learned, sometimes painfully, a set of habits that separate convincing work from anecdotes. Start with a behaviour you can measure precisely, on a dataset, not a single prompt: a circuit found on five prompts is a hypothesis about those prompts. Form hypotheses with the descriptive tools, then test them with causal interventions, and use several corruptions, metrics and ablation types, because each choice isolates a different mechanism and backup mechanisms can hide a component's role.

Beware **interpretability illusions**: a direction that seems to control a concept may act through a different mechanism, especially in subspace interventions (26.13 shows how this happens and how to guard against it). Avoid cherry-picking: report how many features or examples you looked at and how many fit the story. And compare with simple baselines (probes, prompting, a difference-of-means direction) before claiming that an SAE-based method is useful.

## 26.11 A worked investigation: does the model track the speaker's language?

**Question.** A multilingual model answers in the user's language. Is there an internal representation of "language of the conversation", and does the model use it to choose output tokens?

1. **Behaviour first.** Build 2,000 prompts in four languages, with matched content, and measure how often the answer is in the prompt's language. Include conflicting cases (an English instruction quoting a Spanish text) to see which signal wins.
2. **Descriptive tools.**
   - Train linear probes for the language at every layer on residual-stream activations, with a held-out split.
   - Add a control task (random labels assigned per word type) and report selectivity, the gap between the two accuracies (26.3).
   - Expect high accuracy almost everywhere, because language is easy to read from tokens. The interesting result is where a representation of the *conversation's* language exists separately from the *current token's*. Test that on the conflicting cases.
3. **Causal tests.**
   - **Steering.** Take the difference of mean activations between Spanish and English conversations at a middle layer, and add it to English prompts. Does the answer switch language while keeping its content?
   - **Activation patching** (26.5) between matched prompts in two languages localises the layers and positions where patching flips the output language.
   - Use several corruptions and report the share of prompts flipped, not single examples.
4. **Sparse features.** If an SAE is available for that layer, look for latents that fire on the conversation's language rather than on single tokens. Check them causally (ablate, amplify) and compare with the mean-difference direction. If the simple direction steers as well as the SAE latent, say so (26.10).
5. **Report.**
   - The behavioural rates.
   - The probe curves with their controls.
   - The causal effect sizes with intervals over prompts.
   - The failure cases: prompts where steering changed the content, or did nothing.
   - A short statement of what the evidence does and does not show. For example: "a linear direction at layers 14–18 causally controls output language for 87% of prompts; we did not test whether it is the mechanism the model uses without intervention."

The structure transfers to most interpretability questions: behaviour, then correlational evidence, then causal tests with controls, then honest scoping.

## 26.12 Gradient-based attribution

Before mechanistic interpretability, the standard question was simpler: which parts of *this input* made the model produce *this output*? Attribution methods answer it with a score per input feature (a pixel, a token, a tabular column). They remain the most common interpretability tools in applied work and responsible-AI reviews, and interviewers at big tech ask about them, so it is worth knowing how they work and how they fail.

**Saliency** is the gradient of the output with respect to the input, ∂F/∂xᵢ: how much the output would change for a small change in feature i. **Gradient × input** multiplies it by the feature's value. Both are local: they describe the function's slope at the input, not the contribution of the feature. A feature that has saturated a ReLU or a sigmoid has zero gradient however important it is, and a feature in a steep region can have a large gradient while contributing little.

**Integrated gradients** (Sundararajan et al., 2017) fix this by integrating the gradient along a straight path from a baseline input x′ (an all-zero image, a padding-token sequence) to the input x:

$$
\mathrm{IG}_i(x) = (x_i - x_i') \int_0^1 \dfrac{\partial F(x' + \alpha(x - x'))}{\partial x_i}\, d\alpha
$$

Its key property is **completeness**: the attributions sum exactly to F(x) − F(x′), by the fundamental theorem of calculus along the path. So the attributions account for the whole change in output, no more and no less.

A worked example. Take F(x) = x₁x₂ at x = (2, 3) with baseline (0, 0), so F(x) − F(x′) = 6. Along the path the input is (2α, 3α), so ∂F/∂x₁ = x₂ = 3α and ∂F/∂x₂ = x₁ = 2α. Then IG₁ = 2 × ∫ 3α dα = 2 × 1.5 = 3, and IG₂ = 3 × ∫ 2α dα = 3 × 1 = 3. The attributions sum to 6, as completeness requires, and the symmetric function gives equal credit. Gradient × input at x gives (2 × 3, 3 × 2) = (6, 6), which sums to 12: it double-counts, because the gradient at the end of the path is twice its average along the path.

**SHAP** (Lundberg and Lee, 2017) attributes the output to features using Shapley values from game theory: each feature's average marginal contribution over all orders in which features could be added, with "absent" features replaced by values from a background distribution. Shapley values are the unique attribution that satisfies a short list of fairness axioms (efficiency, symmetry, dummy, additivity), and integrated gradients is a path-based relative of them (the Aumann–Shapley value). Exact Shapley values need all 2ⁿ subsets, so SHAP uses sampling or model-specific shortcuts (TreeSHAP for tree ensembles is exact and fast).

**Known failures.** The *baseline* is a modelling choice and changes the answer: a black baseline makes dark pixels unimportant by construction. For language models, "removing" a token has no neutral baseline. *Sanity checks* (Adebayo et al., 2018) showed that some popular saliency methods produced almost the same maps after the model's weights were randomised, which means the maps reflected the input's edges, not the model; any attribution method should pass the randomisation test. Attributions are *local and correlational*: they describe the function around one input and do not say which internal mechanism produced the output. And with correlated features, SHAP splits credit in ways that depend on how absence is simulated. Use attributions to generate hypotheses and to debug individual predictions (a classifier attending to a watermark rather than the object), not as explanations of the model's reasoning.

## 26.13 Causal abstraction, illusions and model diffing

Two research tools answer questions the earlier sections left open. Causal abstraction asks whether a network implements a specific high-level algorithm, and guards against the illusions that subspace interventions can create. Crosscoders and model diffing ask what changed inside a model when it was fine-tuned, which is the interpretability question closest to post-training work.

### Interchange interventions and DAS

A **causal abstraction** hypothesis says that a network implements a simple algorithm with named intermediate variables, such as "compute the sum of the first two numbers, then compare it with the third". It is tested with **interchange interventions**: run the network on input A, but replace the representation of one variable with its value from input B, and check whether the output changes the way the algorithm predicts when that variable is swapped. The share of interventions where it does is the **interchange intervention accuracy**. Activation patching is a special case in which the "variable" is a whole component. **Distributed alignment search (DAS)** (Geiger et al.) learns where the variable lives: it learns a rotation of a layer's activation space and a subspace within it, optimised so that interchange interventions on that subspace match the algorithm. This finds variables that are spread over many neurons, which neuron-level patching misses.

The danger is that optimising a subspace for an intervention can find an effect the model does not use. Makelov et al. (2023) showed how: a learned direction can combine a component that encodes the variable but is not used downstream with one that is used but does not encode it. Patching along that direction activates a dormant pathway and changes the output, so the intervention succeeds and the conclusion ("the variable lives here") is wrong. This is the **interpretability illusion** named in 26.10. Guards against it: check that the found subspace also works for held-out inputs and for interventions it was not optimised on, test whether the effect survives when the subspace is restricted to directions the model actually varies along on real data, and prefer subspaces with an independent justification (a probe, an SAE feature, a known circuit).

### Crosscoders and model diffing

A **crosscoder** is an SAE trained on the activations of two models (or several layers) at once, with one shared set of latents and a separate decoder for each model. Each latent's decoder norms show where it lives: a latent with a large decoder for both models is shared, and a latent whose decoder is near zero for one model exists only in the other. Training a crosscoder on a base model and its chat version is a form of **model diffing**: the chat-only latents are candidates for what fine-tuning added, such as refusal, a persona or a response format. Later work found that the standard L1 training objective can make latents look model-specific when they are not, and proposed checks and alternatives (for example BatchTopK crosscoders). As with every SAE result, confirm candidate latents causally by steering or ablating them in the model where they appear.

## Pitfalls

- Treating probe accuracy as evidence the model uses a feature.
- Using zero-ablation and concluding a component is essential.
- Ignoring redundancy: single ablations under-state what a set of components does.
- Reporting only denoising or only noising when the conclusion depends on the difference.
- Trusting attribution patching for effect sizes where attention saturates.
- Optimising SAEs for reconstruction alone.
- Forgetting that an L1 SAE shrinks every active feature by λ/2.
- Building a circuit story from a handful of prompts.
- Trusting a learned subspace intervention without checking for a dormant pathway.
- Saliency maps without a randomisation sanity check.
- Assuming chains of thought faithfully describe the model's reasoning.

## Interview questions

Questions 1–5 basic, 6–10 intermediate, 11–16 senior.

**1. Explain the residual stream view of transformers.** Each component reads linearly from a shared vector and adds its output back; logits are a sum of contributions; components communicate through the stream, which allows decomposition into paths and circuits. Formally, x_{l+1} = x_l + Σ_h head_{l,h} + MLP_l, so x_L = x₀ plus the sum of every component's output, and the logits W_U·LN(x_L) are approximately a sum of per-component direct effects. The width d is far smaller than the total output of all components, so components use subspaces, and superposition (26.7) packs more into them. Follow-up: "what breaks the linear decomposition?" The final layer norm's scale and the nonlinearities inside MLPs and attention; direct logit attribution freezes the scale.

**2. What is an induction head and how does it work?** A layer-1 head that attends from token A to the position after a previous occurrence of A and copies that token, implementing [A][B] … [A] → [B]. It relies on a layer-0 previous-token head via K-composition: the earlier head writes "my previous token was A" into B's position. The query carries the current token (A), the key carries the previous-token information the earlier head wrote, and the value carries the attended token's identity (B); the OV circuit is a copying circuit with a large positive diagonal in W_U W_OV W_E. In the toy example, a score of 5 for the matching key gives attention 0.98. Induction heads form abruptly early in training, together with a jump in in-context learning, Olsson et al.'s phase change. Follow-up: "how would you find them in GPT-2 small?" Repeated random tokens, a high attention score on the induction offset, and ablation raising loss on the repeat.

**3. How does activation patching work, and how is it different from ablation?** Patching restores one component's clean activation into a corrupted run to test whether it carries the information that restores behaviour; ablation removes a component. Patching is targeted to a specific contrast. Report the normalised effect (m_patched − m_corr)/(m_clean − m_corr) on a logit difference: in 26.6, patching both induction heads gives (0.974 − 0.030)/(0.975 − 0.030) = 0.999. Denoising (clean into corrupted) tests sufficiency; noising (corrupted into clean, like ablation) tests necessity, and redundant heads score high on the first and low on the second. Path patching restricts the patch to one edge of the circuit. Follow-up: "why logit difference rather than accuracy?" It is continuous and responds before the top token flips.

**4. What can go wrong with probing?** Decodable is not used; powerful probes memorise; random features already make many properties decodable. Use linear probes, control tasks, baselines and causal follow-ups. In the check script, a random network's hidden layer already gives 0.62 on a non-linear target that the raw input gives 0.49 on, so a probe score means little without that baseline. Follow-up: "what is selectivity?" Probe accuracy on the real task minus accuracy on a control task with random labels.

**5. What is superposition, and why does it motivate SAEs?** Models represent more sparse features than dimensions as overlapping directions, making neurons polysemantic. It needs three conditions: sparsity (features rarely co-occur, so interference rarely costs), varied importance (unimportant features are dropped, important dense ones get their own dimension), and a nonlinearity to filter interference (a linear model just does PCA). Geometry makes room for it: the number of directions with pairwise |cos| ≤ ε grows like exp(dε²/4), about 28,000 at d = 4,096 and ε = 0.1. SAEs learn an over-complete sparse dictionary to recover those directions. Follow-up: "why are neurons polysemantic then?" Because feature directions are not aligned with neurons, each neuron projects onto many of them.

**6. How would you evaluate a sparse autoencoder?** Sparsity (L0), reconstruction, downstream loss with the SAE spliced in, feature interpretability, and usefulness on concrete tasks compared with baselines. The headline trade-off is downstream loss against L0, compared across SAEs at equal L0; also report dead latents. In the check script, an SAE with near-perfect reconstruction recovered only 3 of 24 true features, which is why reconstruction alone is not the goal. Follow-up: "what baseline?" A linear probe or a difference-of-means direction for the same task; several 2025 studies found they match SAEs on some tasks.

**7. TopK versus L1 SAEs?** TopK fixes the number of active latents directly and avoids shrinkage of active latents caused by the L1 penalty; L1 needs tuning of the penalty and trades sparsity against reconstruction less directly. The shrinkage is λ/2 per active latent: minimising (a − f)² + λf gives f = a − λ/2, so with λ = 0.3 an activation of 1 is reported as 0.85. TopK's costs are a fixed k for every input (some inputs need more features) and more dead latents, handled with an auxiliary loss. Follow-up: "what does JumpReLU do?" A learned threshold per latent with an L0 penalty: no shrinkage above the threshold, and variable sparsity per input.

**8. What is a steering vector? Give an example.** A direction added to the residual stream to change behaviour, x ← x + αv, for example the mean difference between activations on refusing and complying prompts, which can add or remove refusals. The coefficient needs a sweep: plot the intended effect and the loss increase on unrelated text against α, and pick the range where the first is high and the second small. Directional ablation, x ← x − r̂r̂ᵀx, removes a concept everywhere and can be folded into the weights. Follow-up: "why does this matter for open-weight models?" Removing the refusal direction disables refusal training with no fine-tuning.

**9. Why doesn't the logit lens work on early layers, and what fixes it?** Early representations are not in the output basis; the tuned lens learns an affine map per layer, trained to minimise the KL divergence from the final layer's distribution, to decode them. The logit lens is W_U·LN_f(x_l); the tuned lens is W_U·LN_f(A_l x_l + b_l). In the check script's induction model, the lens reads chance accuracy on induction tokens before layer 1 (0.035) and 0.972 after it, locating the computation in one layer. Follow-up: "is the tuned lens's answer what the model 'thinks'?" Only approximately: the affine map is trained, so it can add information the layer does not make easily available.

**10. Is a reasoning model's chain of thought a faithful explanation?** Not necessarily: models can use unstated reasons, and optimisation pressure on the chain of thought can make it less informative. Useful for monitoring, not proof. The evidence is measured: biased few-shot prompts shifted answers while the chain of thought rationalised them (accuracy drops up to about 36%, Turpin et al.), and in a 2025 hint study reasoning models mentioned a hint they used only about 25–39% of the time. Follow-up: "how would you measure faithfulness for your model?" Insert hints and count disclosures, or truncate and corrupt chains of thought and measure how often the answer changes.

**11. Write down the QK and OV circuits of an attention head and say what each does.** The score between query position i and key position j is s_ij = x_iᵀ W_QK x_j/√d_h with W_QK = W_Qᵀ W_K, which decides where to look; the head writes Σ_j A_ij W_OV x_j with W_OV = W_O W_V, which decides what to move. Both are d × d matrices of rank at most d_h, and only the products are meaningful, since any invertible matrix can be moved between W_Q and W_K. Pushing the embeddings through gives vocabulary-level tables: W_Eᵀ W_QK W_E (which token attends to which) and W_U W_OV W_E (what attending to a token does to the logits). Composition is when a later head's W_QK or W_OV reads an earlier head's W_OV output, measured by ‖W_QK² W_OV¹‖_F normalised by the two norms, though that weight-based score can be near the random baseline even for a real circuit (it was in the check script), so confirm with patching. Follow-up: "what does a copying head look like?" A large positive diagonal in W_U W_OV W_E.

**12. Derive why an L1 SAE shrinks features, and how you would fix it.** For one active latent with a unit decoder, the loss in f is (a − f)² + λf for f ≥ 0. The derivative −2(a − f) + λ is zero at f = a − λ/2, so every active latent is reduced by λ/2 and anything below λ/2 is zeroed: soft-thresholding, as in the lasso. With a = 1 and λ = 0.3 the latent is 0.85, and reconstructions are systematically too short. Fixes: TopK (no magnitude penalty, fixed k), JumpReLU (learned threshold with an L0 penalty, used in Gemma Scope), gated SAEs (penalise the gate, not the magnitude), or a post-hoc rescaling of active latents. Follow-up: "why weight the penalty by decoder norm?" Otherwise the SAE shrinks f and grows the decoder to dodge the penalty.

**13. Your attribution-patching scan and real patching disagree about a head. Which do you believe, and why might they differ?** Real patching, because attribution patching is a first-order Taylor estimate, (a_clean − a_corr)ᵀ∇m at the corrupted run, and is wrong wherever the model is non-linear between the two activations. The usual culprits are attention that saturates (restoring one head moves its attention from diffuse to sharp), layer norms whose scale changes, and all-or-nothing components. The trade-off: attribution patching costs two forward passes and one backward pass for every component, against one pass per component for real patching, so use it to shortlist and confirm the top candidates. Follow-up: "can you improve the approximation?" Integrate the gradient along the path from corrupted to clean, as integrated gradients does, at the cost of several backward passes.

**14. Explain integrated gradients and when you would distrust it.** IG integrates the gradient along a straight path from a baseline x′ to the input, IG_i = (x_i − x′_i)∫∂F/∂x_i dα, and satisfies completeness: the attributions sum to F(x) − F(x′). For F = x₁x₂ at (2, 3) with a zero baseline, IG gives (3, 3), summing to 6, while gradient × input gives (6, 6), summing to 12. Distrust it when the baseline is arbitrary (no neutral token for text), when features are correlated, and when the method fails the randomisation sanity check of Adebayo et al. Attributions are local and correlational: they say which inputs mattered here, not how the model computed. Follow-up: "how does it relate to SHAP?" IG is the Aumann–Shapley value along one path; SHAP averages marginal contributions over feature orderings.

**15. What is an interpretability illusion, and how do you guard against it?** A subspace intervention that changes behaviour by activating a pathway the model does not normally use, so the conclusion "the variable lives here" is wrong. Makelov et al. showed a learned direction can mix a dormant component that encodes the variable with an active one that does not, and patching along it succeeds. Guards: test the subspace on held-out inputs and interventions it was not optimised on, restrict it to directions the model varies along on real data, and look for independent support (a probe, an SAE feature, a known circuit). Follow-up: "what is DAS?" Distributed alignment search learns a rotation and subspace so interchange interventions match a hypothesised algorithm, which is exactly the setting where the illusion can arise.

**16. How would you find out what RLHF changed inside a model?** Model diffing. Train a crosscoder on activations of the base and chat models with shared latents and separate decoders, then look at latents whose decoder norm is near zero for the base model: candidates for what post-training added (refusal, persona, format). Validate them causally by steering or ablating in the chat model, and beware that L1 training can make latents look chat-specific when they are not. Cheaper complements: compare probes and steering vectors across the two models, and track a known direction such as refusal. Follow-up: "what if nothing new appears?" Post-training may mostly reweight existing features, which you would see as changed activation frequencies of shared latents.

## Practice

1. Run the check script; then add a third layer and test whether the circuit changes.

<details><summary>Hint and expected result</summary>

Expect the same circuit (a previous-token head feeding induction heads) to form, possibly spread over more heads, with accuracy on repeated tokens still above 0.95. Re-run the ablations: ablating whichever head has the highest previous-token score should hurt induction accuracy most. Expect the weight-based composition scores to stay close to the random baseline, as in the two-layer model; the causal tests are the evidence.

</details>

2. With TransformerLens or nnsight, find induction heads in GPT-2 small using repeated random tokens, and ablate them.

<details><summary>Hint and expected result</summary>

Feed a sequence of random tokens repeated twice. Induction heads attend from each token in the second half to the token after its first occurrence; the strongest are in layers 5–7 (heads such as 5.1, 5.5 and 6.9 are the ones usually reported). Loss on the second half should be far below the first half, and ablating the induction heads should push it back towards the first-half level.

</details>

3. Reproduce activation patching on the IOI task in GPT-2 small; compare logit-difference and accuracy metrics.

<details><summary>Hint and expected result</summary>

The clean logit difference between the indirect object and the subject is about 3.5 on the standard IOI prompts. Patching the residual stream position by position should show the information moving from the subject position in early layers to the final position in later layers, with name-mover heads in layers 9–10. Accuracy will show the same picture with much coarser steps.

</details>

4. **Research exercise.** Train TopK SAEs with k ∈ {8, 16, 32, 64} on one residual-stream layer of a small open model (a few million tokens). Plot unexplained variance and downstream loss increase against L0, and label 30 random features per SAE by hand. Does interpretability change with k?

<details><summary>Hint and expected result</summary>

Expect unexplained variance and the downstream loss increase to fall steadily with k, and the share of features you can label confidently to fall as k grows, with more features that fire densely and resist a single description. Also count dead latents per SAE. Report the interpretability judgement with an interval over the 30 features, and ideally have a second person label blind.

</details>

5. Compute a refusal direction for a small open chat model from 100 harmful and 100 harmless prompts; measure refusal rates when removing and adding it.

<details><summary>Hint and expected result</summary>

Pick the layer and position (usually the last token of the prompt, in a middle layer) where the direction best separates the two sets on held-out prompts. Expect directional ablation to cut refusals on harmful prompts to a small fraction, and adding the direction to make the model refuse many harmless prompts. Check fluency on unrelated text after the edit.

</details>

6. Implement integrated gradients for a small text classifier and run the randomisation sanity check.

<details><summary>Hint and expected result</summary>

Use the embedding of a padding token as the baseline, 50 steps along the path, and check completeness: the attributions should sum to the logit change within a few per cent. After randomising the classifier's top layer, the attributions should change substantially; if they do not, they reflect the input rather than the model.

</details>

## Read next

- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021).
- Olsson et al., "In-context Learning and Induction Heads" (2022).
- Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" (2022).
- Meng et al., "Locating and Editing Factual Associations in GPT" (ROME, 2022).
- Elhage et al., "Toy Models of Superposition" (2022); Bricken et al., "Towards Monosemanticity" (2023); Templeton et al., "Scaling Monosemanticity" (2024).
- Gao et al., "Scaling and evaluating sparse autoencoders" (2024); Rajamanoharan et al. on gated and JumpReLU SAEs (2024); Lieberum et al., "Gemma Scope" (2024).
- Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in Language Models" (2025).
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (2024).
- Zhang and Nanda, "Towards Best Practices of Activation Patching in Language Models" (2023).
- Turpin et al., "Language Models Don't Always Say What They Think" (2023); Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning" (2023); Chen et al., "Reasoning Models Don't Always Say What They Think" (2025).
- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017); Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions" (2017); Adebayo et al., "Sanity Checks for Saliency Maps" (2018).
- Geiger et al. on causal abstraction and distributed alignment search; Makelov et al., "Is This the Subspace You Are Looking For?" (2023).
- Lindsey et al., "Sparse Crosscoders for Cross-Layer Features and Model Diffing" (2024).
- Links to the core papers are in the Key concepts section at the top of this chapter.

## Done when

- Without notes, write the residual stream as a running sum, the logits as W_U·LN(x_L), and W_QK = W_Qᵀ W_K and W_OV = W_O W_V, saying what each reads and writes, in 5 minutes.
- Work the A B C A induction example by hand: attention on the matching key is e⁵/(e⁵ + 3) ≈ 0.98.
- Compute the normalised patching effects from the 26.6 table: 0.731, 0.619 and 0.999, and explain why the single effects sum to more than 1.
- State denoising versus noising and which one tests necessity.
- Write the attribution-patching estimate and name two situations where it fails.
- Write the SAE encoder, decoder and loss from memory, derive f = a − λ/2, and compute it for a = 1, λ = 0.3: answer 0.85.
- Derive N ≈ exp(dε²/4) and evaluate it for d = 4,096, ε = 0.1: about 28,000.
- Compute integrated gradients for F = x₁x₂ at (2, 3) with a zero baseline: (3, 3), and gradient × input: (6, 6).
- Find an induction circuit in a small model with attention patterns, ablations and patching, and run a probe with a control task.
- Explain an interpretability illusion and one guard against it, and describe how a crosscoder diffs a base and a chat model.

---

[← 25 · RAG and agent evaluation](25-rag-and-agent-evaluation.md) · [Contents](../README.md#contents) · [27 · Alignment and safety research →](27-alignment-and-safety-research.md)
