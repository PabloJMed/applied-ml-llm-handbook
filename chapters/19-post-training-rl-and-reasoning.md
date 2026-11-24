# 19 · Post-training II: reinforcement learning and reasoning models

Part V · Post-training. Assumes Chapter 18 (PPO, advantages, clipping, KL, reward over-optimisation) and Chapter 13 (test-time compute, pass@k). Chapter 33 extends the same methods to multi-step agents; Chapter 27 covers reward hacking as an alignment problem.

## Why it matters

From late 2024 the most important advances in LLMs came from reinforcement learning on tasks whose answers can be checked: maths with known answers, code with tests, puzzles and tool tasks with verifiable end states. Models trained this way think at length before answering, check and revise their work, and scale their accuracy with the inference compute they are given. DeepSeek-R1's open report (2025) showed the recipe could be reproduced; since then, RL has become a large and growing share of post-training compute. For a research scientist, this is the most active area of the field: the algorithms (GRPO and its fixes), reward design and verifiers, curricula, the debate about whether RL teaches new reasoning or sharpens what pretraining already gave, and how to spend inference compute well.

Numbers in this chapter come from `code/ch19_check.py` (NumPy, under a second).

The chapter starts with what changed in 2024 to make RL for reasoning work (19.1), then takes the components of a run in turn: the verifier that defines the reward (19.2), GRPO, the algorithm most open recipes use (19.3), the biases found in it and their fixes (19.4), and the choice of training problems (19.5). Section 19.6 asks the research question behind it all: does RL teach new reasoning, or only surface what the base model could already do? Sections 19.7 to 19.9 cover process rewards, ways to spend inference compute, and distilling reasoning into small models. The chapter ends with the systems side (19.10), a worked run (19.11) and what is known about predicting RL results at scale (19.12).

The key difference from Chapter 18 is the reward. A verifier is cheap, exact and hard to fool, so the policy can be optimised far harder than against a learned reward model, for thousands of steps. Most of the chapter's techniques are about using that freedom well: choosing problems of the right difficulty, assigning credit fairly across long responses, and keeping generation, which dominates the cost, fast.

An interviewer for a reasoning or RL team expects three things. First, the objective written out: the full GRPO loss with its normalisations, and where each 2025 fix (Dr. GRPO, DAPO, GSPO, CISPO) changes it. Second, an opinion backed by evidence on whether RL teaches new reasoning, with the experiment that would settle it. Third, the cost model of a run: tokens per step, the split between generation and training, and what the dashboard should show. Sections 19.3, 19.6 and 19.11 are written to those three questions.

## Key concepts

- **RL with verifiable rewards (RLVR)**: reward from a programmatic check of the final answer. [Tülu 3 (RLVR)](https://arxiv.org/abs/2411.15124), [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
- **GRPO**: policy optimisation with group-relative advantages and no value model. [DeepSeekMath](https://arxiv.org/abs/2402.03300).
- **Fixes to GRPO**: [Dr. GRPO (Understanding R1-Zero-like training)](https://arxiv.org/abs/2503.20783), [DAPO](https://arxiv.org/abs/2503.14476).
- **Process reward models (PRMs)**: score each reasoning step, not only the answer. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), [Math-Shepherd](https://arxiv.org/abs/2312.08935).
- **Test-time scaling**: self-consistency, best-of-n, search, longer thinking. [Self-consistency](https://arxiv.org/abs/2203.11171), [Snell et al.](https://arxiv.org/abs/2408.03314), [s1 (budget forcing)](https://arxiv.org/abs/2501.19393).
- **Does RL add new capability?** [Yue et al.](https://arxiv.org/abs/2504.13837).
- **Scaling RL in practice**: [Kimi k1.5](https://arxiv.org/abs/2501.12599), [ScaleRL (The Art of Scaling RL Compute)](https://arxiv.org/abs/2510.13786).
- **Sequence-level and clipped-weight objectives**: [GSPO](https://arxiv.org/abs/2507.18071), [CISPO (MiniMax-M1)](https://arxiv.org/abs/2506.13585).
- **Training–inference mismatch**: [truncated importance sampling (Yao et al.)](https://fengyao.notion.site/off-policy-rl), [Defeating nondeterminism in LLM inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
- **Rewards beyond verifiable tasks**: [Rubrics as Rewards](https://arxiv.org/abs/2507.17746).
- **On-policy distillation**: [Thinking Machines (2025)](https://thinkingmachines.ai/blog/on-policy-distillation/), [GKD](https://arxiv.org/abs/2306.13649).

## 19.1 What changed

Earlier models were taught to reason by imitating written solutions, and they learned the form of reasoning more than its substance. The breakthrough of 2024 was to let the model find its own solutions, rewarding only correct final answers, at large scale. Long chains of thought, self-checking and backtracking then appeared without being demonstrated.

Earlier attempts to teach reasoning through supervised chains of thought or learned process rewards helped but did not scale cleanly. The recipe that worked is simple:

1. Start from a strong base model (pretraining and mid-training on maths and code matter a lot).
2. Optionally, a small SFT "cold start" on long, well-formatted reasoning traces.
3. Run RL on many prompts with checkable answers, rewarding correctness (and format), with a policy-gradient method such as GRPO.
4. Distil the resulting behaviour into smaller models, and mix reasoning data with general post-training (Chapter 18).

DeepSeek-R1-Zero went straight from base model to RL with no SFT and still learned long reasoning, self-verification and backtracking (the "aha moment" of the report), though with readability problems that the cold-start SFT fixed in R1. Response length grew steadily during training: the model learned to use more tokens because more thinking earned more reward.

**The full R1 pipeline** had four stages, and each fixed a problem of the one before. (1) A cold-start SFT on a few thousand long, readable reasoning traces, so that RL starts from a model that writes legible chains of thought. (2) Reasoning RL with GRPO on maths, code and logic, rewarded for correctness and format, plus a **language-consistency reward** (the share of the chain of thought written in the target language), added because R1-Zero mixed languages mid-thought; the report notes it cost a little accuracy and bought readability. (3) **Rejection-sampled SFT**: sample many responses from the RL checkpoint, keep the correct and readable ones (about 600k reasoning samples), add about 200k non-reasoning samples (writing, factual QA, self-cognition), and fine-tune the base model again on these 800k. (4) A final RL stage over all scenarios, with rule-based rewards for reasoning and reward models for helpfulness and harmlessness. The same 800k samples were then used to distil the smaller models (19.9).

## 19.2 Verifiers and reward design

In RL with verifiable rewards, the verifier is the entire specification of what the model should learn. Anything it fails to check, the policy is free to exploit, so reward design is mostly about closing loopholes: answers in the wrong format, tests that can be gamed, or partial credit that rewards the wrong behaviour.

The reward is the specification. Typical verifiers:

| Domain | Verifier | Ways it gets gamed |
|---|---|---|
| Maths with a final answer | Extract the answer (for example the last boxed expression), normalise, compare symbolically | Lenient matchers accept wrong forms; answers guessed without valid reasoning |
| Code | Run hidden unit tests in a sandbox | Special-casing tests, exiting early with success codes, editing the tests in agentic setups |
| Instruction constraints | Programmatic checks (word count, JSON validity, keywords) | Satisfying the letter, not the spirit |
| Tool and agent tasks | Check the final environment state (Chapter 25) | Shortcuts the environment allows but the task forbids |
| Open-ended tasks | LLM judge with a rubric | Judge biases (length, style), sycophancy towards the judge |

Practical rules:

- **Harden verifiers as carefully as the policy.** Sample rewarded outputs regularly and read them. Every verifier gets exploited eventually.
- **Format rewards are small and separate**: reward the expected thinking/answer structure lightly so it cannot dominate correctness.
- **Partial credit** (fraction of tests passed) gives denser signal but rewards partial hacks; binary rewards are harder to game.
- **Decontaminate prompts** against evaluation sets: RL on contaminated prompts inflates reasoning benchmarks fast.

**Beyond verifiable tasks: rubrics as rewards.** Most useful work (medical advice, research summaries, writing) has no single checkable answer. A practical middle ground between a verifier and an opaque reward model is a **rubric per prompt**: a checklist of weighted criteria ("mentions the drug interaction", "recommends seeing a doctor for symptom X", "does not invent a dosage"), written by experts or generated by a strong model from a reference answer, and scored by an LLM judge. The reward is the weighted share of criteria met. Gunjal et al. (2025) found rubric rewards beat a judge giving a single Likert score, and let smaller judges track human preferences more closely. RLVR is the special case with one criterion, "the answer matches". The risks are the judge's: verbosity, criteria satisfied in letter only, and rubric items the policy learns to recite. Audit high-reward samples as you would for any verifier.

## 19.3 GRPO in detail

PPO needs a value model as large as the policy to estimate how good each state is, which doubles memory and adds its own training problems. GRPO removes it with a simple idea: sample several answers to the same prompt, and judge each one against the others. The group's average reward is the baseline, and each answer's advantage is how much better or worse it did than its siblings.

For each prompt, sample a group of G responses (8 to 64), score each with the verifier, and use the score relative to the group as the advantage of every token in that response:

$$
A_i = \dfrac{r_i - \mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)}
$$

No value model is needed: the group is the baseline. The objective is PPO's clipped surrogate (Chapter 18) applied per token with this advantage, often with a KL penalty to the reference model.

For rewards [1, 0, 0, 1, 1, 0, 0, 0], the advantages are +1.21 for correct and −0.72 for wrong responses (mean 0.375, sample standard deviation 0.518).

### The full objective

Written out, as in DeepSeekMath, GRPO maximises, for a question q with G sampled outputs o₁ … o_G:

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\left[\dfrac{1}{G}\sum_{i=1}^{G}\dfrac{1}{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}(\theta)\right]
$$

where each token contributes a clipped surrogate minus a KL term:

$$
\ell_{i,t} = \min\!\left(\rho_{i,t}\hat{A}_{i},\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\,\hat{A}_{i}\right) - \beta\,D_{i,t}
$$

The symbols: |o_i| is the length of output i in tokens; Â_i is the group-relative advantage above, the same for every token of output i; ρ_(i,t) is the per-token probability ratio between the policy being trained and the policy that sampled the data,

$$
\rho_{i,t} = \dfrac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})}
$$

and D_(i,t) is Schulman's k3 estimate of the KL to the reference model (Chapter 18.4), computed on the sampled token with r = π_ref/π_θ:

$$
D_{i,t} = r_{i,t} - \log r_{i,t} - 1
$$

**Two tokens by hand** (ε = 0.2, β = 0.04, the KL coefficient DeepSeekMath used). A token of a correct response (Â = +1.21) whose ratio has risen to 1.3, with π_ref/π_θ = 0.9: the surrogate is min(1.3 × 1.21, 1.2 × 1.21) = min(1.573, 1.452) = 1.452, so the clip has removed any incentive to push further; the KL term is 0.04 × (0.9 − log 0.9 − 1) = 0.04 × 0.0054 = 0.0002. A token of a wrong response (Â = −0.72) whose ratio has fallen to 0.7: min(0.7 × −0.72, 0.8 × −0.72) = min(−0.504, −0.576) = −0.576, clipped from below in the same way.

**Where the KL sits.** In InstructGPT-style PPO (Chapter 18.4) the KL penalty is subtracted from the *reward* at every token, so it flows into the advantages and the value targets. GRPO adds it directly to the *loss*, outside the advantage: the group-relative advantage contains only the verifier's reward, and the KL acts as a separate regulariser on every token. That keeps the advantage interpretable (better or worse than siblings at the task), but it is not free of subtleties: the gradient of a sampled KL estimate placed in the loss is not automatically the gradient of the KL, and 2025 analyses pointed out that k3-in-the-loss optimises a slightly different divergence than intended. With β small or zero, as in most long-reasoning recipes (19.4), the question matters little.

**What each normalisation does.** Three choices in the formula turn out to shape behaviour, and the fixes of 19.4 each change one of them. The 1/|o_i| averages each response's loss over its own length before averaging over the group, which creates the length bias. The division by the group's standard deviation inside Â_i re-weights prompts by how consistent their results are. And the per-token ratio and clip treat each token as its own action even though the reward belongs to the whole sequence. DAPO replaces 1/G Σ 1/|o_i| by one average over all tokens in the batch, 1/Σ|o_i|; Dr. GRPO replaces it by a constant (the maximum length) and drops the standard deviation; GSPO and CISPO change the ratio.

**Why groups work.** A value model for long reasoning must predict, from a partial chain of thought, the chance of eventual success: hard to learn and memory-hungry. A group of samples estimates the prompt's baseline directly, at the price of more generation.

## 19.4 Known biases and the fixes

Careful analysis in 2025 found that the original GRPO objective contains normalisation choices that looked harmless and turned out to shape behaviour, most visibly the length of responses, and that several engineering choices matter as much as the algorithm. Understanding them matters both for running RL and for interpreting published results, since several "emergent" behaviours turned out to be partly artefacts of the objective.

**Length normalisation bias.** Original GRPO averages the loss over the tokens of each response, then over responses: the 1/|o_i| in the objective of 19.3. Each token's weight is A_i / |o_i|. The example group has eight responses with rewards [1, 0, 0, 1, 1, 0, 0, 0] and lengths [200, 900, 1,500, 300, 250, 400, 1,200, 2,000] tokens, 6,750 in all; the three correct ones are short. For a correct 200-token answer that is +6.04e-3 per token; for wrong answers of 1,500 and 2,000 tokens it is −0.48e-3 and −0.36e-3. Longer wrong answers are penalised less per token, so the model drifts towards longer outputs when it is wrong. Normalising by a constant (Dr. GRPO) or averaging over all tokens in the batch (DAPO's token-level loss) gives every token the same weight: +0.18e-3 for correct tokens (1.21/6,750) and −0.11e-3 for wrong ones (−0.72/6,750), whatever the length.

The same numbers per response make the incentive plain. Under sequence averaging, every response carries the same total weight whatever its length, A_i/G: +0.151 for each correct response and −0.091 for each wrong one, so a 2,000-token wrong answer is penalised no more in total than a 400-token one and much less per token. Under token averaging, the total weight grows with length: −0.215 for the 2,000-token wrong answer against −0.043 for the 400-token one. Rambling on a failed attempt stops being cheap.

**Standard-deviation normalisation.** Dividing by the group's standard deviation up-weights prompts that are nearly always solved or nearly always failed (small std), which are not necessarily the most informative. Dr. GRPO drops the division; others keep it. The choice changes which prompts dominate learning.

**Zero-signal groups.** If every response in a group gets the same reward, all advantages are zero and the prompt contributes nothing. With 8 samples, a prompt with a 2% or 98% pass rate gives no signal 85% of the time; one at 50% almost never. **Dynamic sampling** (DAPO) keeps sampling until the batch is filled with groups that have mixed results.

![Probability that a group of G samples is all correct or all wrong (and so gives zero advantage) as a function of the prompt's pass rate](images/19-fig-1.png)

*Probability that a group of G samples is all correct or all wrong (and so gives zero advantage) as a function of the prompt's pass rate. Larger groups shrink the dead zone at both ends but never remove it.*

Larger groups help at the edges. A prompt with a 5% pass rate gives no signal 66% of the time with G = 8 and 44% with G = 16. One at 20% drops from 17% to 3%. The cost is twice the generation per prompt, so in practice teams use G = 8–16 together with a difficulty filter and dynamic sampling.

**Entropy collapse and clip-higher.** The policy's entropy is measured as the mean token entropy over the sampled responses: at each generated position, the entropy of the full next-token distribution, averaged over all positions in the batch,

$$
\mathcal{H} = -\dfrac{1}{\sum_{i}|o_i|}\sum_{i,t}\ \sum_{v}\pi_{\theta}(v\mid s_{i,t})\log\pi_{\theta}(v\mid s_{i,t})
$$

where v runs over the vocabulary and s_(i,t) is the prompt plus the first t − 1 tokens of output i. A position where the model spreads its probability 0.5 / 0.3 / 0.2 over three tokens has entropy 1.03 nats; 0.9 / 0.05 / 0.05 gives 0.39; 0.99 / 0.005 / 0.005 gives 0.06. RL quickly reduces this number: the policy commits to the strategies that already work and stops exploring.

Clipping is part of the reason, because the clip bounds the *ratio*, a multiplicative change. With symmetric clipping at 1 ± 0.2, a token with probability 0.01 can rise to at most 0.012 per update, an absolute gain of 0.002, while a token at 0.5 can gain 0.1. Rare tokens, which are exactly the unusual moves ("wait, let me recheck") that exploration needs, are the slowest to grow, and confident tokens are reinforced freely. DAPO's **clip-higher** raises the upper bound (to 1.28), so the rare token can reach 0.0128 and entropy stays up. Other tools: entropy bonuses, a higher sampling temperature, and removing the KL penalty (DAPO removes it, since the policy is supposed to move far from the reference in long-reasoning RL).

**Overlong responses.** Responses cut at the length limit get no reward though their reasoning may be fine, which adds noise. DAPO masks truncated samples from the loss or applies a soft penalty as responses approach the limit.

**Where the importance ratio lives.** GRPO and PPO apply the ratio π_new/π_old and the clip per *token*, while the reward and advantage belong to the whole *sequence*. Two 2025 objectives move them into line:

- **GSPO** (Qwen, 2025) uses one sequence-level ratio, the geometric mean of the per-token ratios (exp of the mean per-token log-ratio), and clips whole responses. The length normalisation keeps the ratio in a sane range, unlike the raw product of section 18.4. It was motivated by collapses in large mixture-of-experts RL, where about a tenth of the routed experts can change for the same tokens after one update, making per-token ratios very noisy. The Qwen team report it made MoE RL stable without "routing replay" (forcing the old routing during the update) and that it tolerates small numerical differences between the inference engine and the trainer.
- **CISPO** (MiniMax-M1, 2025) keeps a REINFORCE-style objective, (stop-gradient of the clipped weight) × advantage × ∇log π, and clips only the importance *weight*. Under PPO-style clipping, rare "fork" tokens ("wait", "recheck") whose ratio jumps above 1 + ε get no gradient at all; under CISPO they still learn, with a bounded weight.

Both are answers to the same question from section 18.4: how to keep importance correction stable when responses are thousands of tokens long.

Other algorithms in the same family: **RLOO** (leave-one-out baseline within the group), **REINFORCE++** and various "critic-free" PPO variants. PPO with a value model has not disappeared either: some 2025 reasoning recipes (for example VAPO) made it competitive again by fixing value-model initialisation and the GAE settings for long responses that Chapter 18.4 discusses. Treat claims that one variant is better with care: reported differences often come from the implementation details above, from learning rates, or from the prompt set.

## 19.5 Data and curriculum

RL only learns from problems where the model sometimes succeeds and sometimes fails. Problems it always solves, or never solves, give every sample the same reward and therefore no learning signal. Choosing and updating the training problems is as important as the algorithm.

- **Difficulty**: keep prompts whose pass rate under the current policy is neither 0 nor 1 (often 10–90%). Re-estimate as the policy improves; prompts that become easy stop teaching.
- **Coverage**: maths of several levels, code in several languages, logic puzzles, STEM questions with short answers, instruction constraints. Narrow prompt sets give narrow gains.
- **Verifiability first**: drop prompts whose reference answers are wrong or ambiguous (a surprising fraction in scraped maths sets); these teach the model to fit noise.
- **Curricula**: increase difficulty or maximum response length in stages; long-context reasoning is expensive, so many runs grow the length limit over training.

## 19.6 Does RL teach new reasoning?

RL raises the share of correct answers, but it is a fair question whether it creates new abilities or only makes the model choose, more reliably, solutions it could already produce occasionally. The answer decides how far RL alone can push a model, and the evidence is more nuanced than either side's headline.

Yue et al. (2025) compared base and RL-trained models with pass@k for large k. RL models win clearly at k = 1, but base models often catch up or win at k = 256: RL mostly **sharpens** the distribution towards solutions the base model could already sample, and can **narrow** it, losing rare solutions. Others argue that longer RL, larger models or more diverse prompts do extend the frontier, and that distillation from a stronger model genuinely adds new capability to a small one. The debate is about evidence, so you should be able to produce some.

The counter-evidence has names. NVIDIA's **ProRL** (2025) ran RL for much longer than typical academic runs (thousands of steps) on diverse tasks, with a KL penalty and periodic resets of the reference policy to keep training stable, and reported pass@k gains at large k on tasks where the base model rarely or never succeeded, the pattern sharpening alone cannot produce. Chu et al. (2025, "SFT Memorizes, RL Generalizes") compared SFT and RL on the same rule-based card and navigation tasks and found that RL-trained models transferred to unseen rule variants while SFT-trained ones mostly memorised the training rules, though SFT was still useful to stabilise the output format before RL. Read together, the evidence suggests that short RL on a narrow prompt set mostly sharpens, while long, diverse, exploration-preserving RL can extend what the model finds, with the base model's pretraining setting how far.

The check script builds a toy version: 200 problems, six solution strategies; one common strategy works on 75% of problems, and the remaining 25% need a rare strategy the base policy picks only 1% of the time. GRPO-style updates on a shared policy:

| | pass@1 | pass@64 | Entropy | pass@64 on the hard 25% |
|---|---|---|---|---|
| Base policy | 0.337 | 0.910 | 1.03 | 0.402 |
| After RL | 0.465 | 0.763 | 0.05 | 0.032 |

pass@1 rises; pass@64 falls, because the rare strategy, reinforced too seldom to compete, is squeezed out. This is entropy collapse in miniature. The lesson for practice: **report pass@k curves, not only pass@1**, monitor entropy, and keep exploration alive (clip-higher, temperature, diverse prompts) if you want RL to find new solutions rather than only sharpen old ones.

![pass@k against k for the base and RL-trained policies in the toy simulation](images/19-fig-2.png)

*pass@k against k for the base and RL-trained policies in the toy simulation. The curves cross: RL improves the first sample and loses the tail, because the rare strategy that solves the hard problems was squeezed out.*

**Reading a pass@k plot.**

- **The k = 1 end** measures how reliably the model finds a solution it knows.
- **The large-k end** measures what it can find at all.
- **Where the curves sit.** An RL run that only sharpens the policy lifts the left end and leaves the right end unchanged or lower. An RL run that genuinely extends the model's reach lifts both ends.

Compute pass@k with the unbiased estimator from n ≥ k samples (Chapter 13), on held-out problems, at the same temperature for both models. Comparing a base model at temperature 1.0 with an RL model at 0.6 is a common way to fool yourself.

**The decisive experiment.** To test whether a run taught something new, select held-out problems the base model fails at large k (say 0 of 1,024 samples at temperature 1.0), sample the RL model at the same temperature and budget, and verify that the solutions it finds are correct by a strict checker and not guessed. A non-trivial pass rate on that set is evidence of extension; zero, with gains only on problems the base model could already solve, is evidence of sharpening. Report it per domain and with confidence intervals, since a handful of lucky guesses on short-answer problems can fake it (multiple-choice and small-integer answers are the usual culprits).

## 19.7 Process supervision

A reward on the final answer says nothing about which steps were good. A model that reaches the right answer through a wrong argument is rewarded, and a model that makes one slip in a long correct argument is not. Process supervision scores the intermediate steps, at the cost of needing a way to judge them.

An outcome reward model (ORM), or a verifier, judges the final answer. A **process reward model (PRM)** scores each step of the reasoning. Lightman et al. (2023) trained a PRM on 800k human step labels (PRM800K) and found that, for selecting the best of many solutions, process supervision beat outcome supervision on competition maths. Math-Shepherd labelled steps automatically: a step's label is the fraction of continuations from that step that reach the right answer (a Monte Carlo estimate of its value).

PRMs are used for: reranking samples, guiding search (beam search over steps, tree search), and dense rewards in RL. Their problems: human step labels are expensive; automatic labels are noisy; and PRMs used as RL rewards get hacked (the policy produces steps that look good to the PRM). R1-style training used outcome rewards only, and most RLVR pipelines still do; PRMs are more common at inference.

## 19.8 Spending inference compute

A model's accuracy can be raised after training by spending more compute per question: sampling several answers, checking them, or searching. The strategies differ in what they need (a verifier, a reward model, or nothing at all) and in how quickly their gains flatten out.

| Strategy | How | Needs |
|---|---|---|
| Self-consistency | Sample n reasoning paths; majority vote on the final answer | Answers that can be compared |
| Best-of-n | Sample n; pick the highest-scoring by a verifier or reward model | A good verifier |
| Search with a PRM | Beam or tree search over reasoning steps | A PRM; more engineering |
| Sequential revision | Answer, critique, revise | A model that can find its own errors |
| Longer thinking | Train the model to think longer; control the budget | RL-trained reasoning model |

**Self-consistency** works when the correct answer is the most common one. With a 30% single-sample accuracy and wrong answers spread over five options, majority voting over 25 samples reaches 70%; at 50% accuracy it reaches 99%. But if one wrong answer is more popular than the right one, voting makes things worse: with a single wrong answer that takes the other 70%, the majority over 25 votes is right only 2% of the time. Voting amplifies whatever the model believes most.

![Accuracy of the majority answer as more samples are voted on](images/19-fig-3.png)

*Accuracy of the majority answer as more samples are voted on. When the right answer is the plurality, voting climbs towards 100%; when a single wrong answer is more popular, voting drives accuracy towards zero.*

The practical check before adding voting is cheap. Sample 20–40 answers on a few hundred dev problems and look at the answer histograms. If the right answer is usually the most common one, voting will help, and the curve tells you how many samples are worth paying for. If a wrong answer is often the plurality (a common misconception, an ambiguous question), use a verifier or reward model to pick instead.

**Budget forcing** (s1, 2025) controls the thinking length at inference: stop thinking early by forcing the end-of-thinking token, or extend it by suppressing that token and appending "Wait". With only 1,000 curated SFT examples, accuracy rose with the forced budget. Reasoning models also **overthink**: they spend thousands of tokens on easy questions. Length penalties and adaptive budgets are an active area.

### Hybrid thinking and effort control

Products cannot afford thousands of thinking tokens for "what is the capital of France", so 2025–26 models expose the thinking budget as a control. **Hybrid models** are trained to work in both modes: Qwen3, for example, fused a thinking and a non-thinking mode into one model during post-training, switched by a flag in the chat template, with a thinking budget that cuts the chain of thought off and forces an answer when it is exhausted. Commercial APIs expose the same idea as a reasoning-effort setting or a token budget for thinking. Training for it means mixing data from both modes (thinking traces and direct answers for the same kinds of prompts) and, in RL, sampling with different budgets so the model learns to produce its best answer within each.

Two practical consequences for evaluation and serving. First, **decoding settings differ**: reasoning models are usually sampled at moderate temperature (DeepSeek recommended about 0.6 for R1) rather than greedily, because greedy decoding of long chains of thought tends to loop and repeat. Second, **cost scales with thinking tokens**, which are billed and add latency like any other output tokens: a model that thinks for 4,000 tokens before a 200-token answer costs about 20 times the output tokens of a direct answer, so effort should be routed by difficulty (Chapter 21 covers the serving side).

## 19.9 Distilling reasoning

RL is expensive and works best on large models. Small models can acquire much of the same ability far more cheaply by training directly on the reasoning traces of a large one. For models below a certain size, distillation consistently beats running RL on the small model itself.

The cheapest way to give a small model reasoning ability is SFT on reasoning traces from a stronger model. DeepSeek released distilled models trained on about 800k samples from R1 that outperformed the same small models trained with RL directly. s1 and LIMO showed that around a thousand carefully chosen, hard, verified traces already elicit much of the behaviour from a strong base model. The emerging rule: **distil into small models, run RL on large ones**, then distil again. Quality filters on traces matter: verify final answers, remove repetitive or incoherent thinking, prefer diverse problems.

**On-policy distillation.** SFT on teacher traces is off-policy: the student learns from states the teacher visits, not the ones it reaches itself (the exposure-bias problem of Chapter 18). On-policy distillation samples responses from the *student* and asks the teacher to score every token, typically with the per-token reverse KL between student and teacher as a dense negative reward. It combines RL's on-policy sampling with SFT's dense signal: one bit of reward per episode becomes a target for every token. The Qwen3 report found it beat RL on the same student (74.4% against 67.6% on AIME'24) at about a tenth of the GPU hours, and Thinking Machines (2025) showed it also restores instruction-following lost after fine-tuning on new domain documents. It needs a teacher that shares the student's tokenizer, or a mapping between vocabularies, and access to the teacher's log-probabilities. The 2025–26 open-model recipes increasingly use it as the last post-training stage.

## 19.10 Systems for RL

A reasoning RL run spends most of its time generating long responses, not computing gradients. The engineering problem is therefore keeping inference fast inside a training loop, with the policy's weights changing at every step.

RL for reasoning is dominated by **generation**: sampling long responses for many prompts takes most of the time. Practical consequences:

- Rollouts run in a fast inference engine (vLLM, SGLang), training in a training framework; weights are synced after each update.
- **Asynchronous RL** overlaps generation and training, which makes samples slightly off-policy; importance weights or bounded staleness keep this in check.
- **Training–inference mismatch**: the inference engine and the trainer compute slightly different token probabilities (different kernels, precision, batch composition), so "on-policy" RL is quietly off-policy. The PPO ratio then mixes real policy change with numerical noise, which can destabilise long runs; section 18.4 shows how fast small per-token differences compound over thousands of tokens. Fixes, roughly in order of adoption:
  - recompute the old log-probabilities in the trainer rather than trusting the engine's;
  - **truncated importance sampling** (Yao et al., 2025): weight each token's gradient by min(π_trainer/π_engine, C), a one-sided cap that trades a little bias for bounded variance, which also makes quantised (8-bit) rollouts usable;
  - keep the output head and its softmax in fp32 (ScaleRL found this raised the final reward);
  - **batch-invariant kernels**, which give identical numerics whatever the batch size (Thinking Machines, 2025), for bitwise on-policy RL at a throughput cost;
  - monitor the gap: log the per-token KL or ratio statistics between engine and trainer, because a growing gap often precedes a collapse.
- Long responses create imbalanced batches; pack and sort by length.
- Open frameworks: verl, OpenRLHF, TRL, and others (Chapter 32).

## 19.11 A worked RLVR run

Plan a run for a 30B-class model on maths and code with verifiable rewards.

**Prompt pool.** Start with 200k scraped maths problems and 50k coding tasks with tests.

1. **Clean the answers.** Drop maths items whose reference answer fails a parser or disagrees with a strong model's majority vote. Then check a sample of the disagreements by hand, because a surprising share of them are wrong references.
2. **Estimate pass rates.** Sample each prompt 8 times with the starting policy and record the pass rate.
3. **Filter by difficulty.** Keep prompts between 10% and 90%. The kept pool is often a third of the original or less.

**Group size and batch.** Use 512 prompts per step and G = 16 samples each, with a response limit of 8k tokens and an average near 4k.

- **Generation per step.** That is 512 × 16 × 4,000 = 32.8M tokens.
- **Where the time goes.** Generation dominates the wall-clock time, so the inference engine's throughput sets the pace of the run. The arithmetic below shows when and why.
- **Zero-signal groups.** With G = 16, a prompt that has drifted to a 5% or 95% pass rate is still dead 44% of the time. Re-estimate pass rates every few hundred steps and refill the pool.

**Cost of one step.** Take a dense 30B model on 256 H100-class GPUs. The training side processes every generated token three times: a forward and backward pass for the update (6N FLOPs per token), a forward pass of the frozen reference model for the KL (2N), and a forward pass to recompute the old policy's log-probabilities in the trainer (2N, 19.10). That is 10N × 32.8M = 9.8 × 10¹⁸ FLOPs, 97 s at 40% MFU. Generation needs only 2N per token, but decoding is memory-bound and runs at low utilisation; its speed is what you must measure. A roofline estimate for this model (64 layers, 8 key–value heads, 262 KB of KV cache per token in bf16, average context about 2,500 tokens) gives roughly 600 tokens/s per GPU with the whole model on one 80 GB GPU, where the cache has room for only about a dozen sequences, and about 4,000 with tensor parallelism over four GPUs, where it holds hundreds.

| Decode throughput per GPU | Generation per step | Step time | Generation share | 1,000 steps |
|---|---|---|---|---|
| 1,000 tokens/s | 128 s | 225 s | 57% | 63 h, 16.0k GPU-hours |
| 2,000 tokens/s | 64 s | 161 s | 40% | 45 h, 11.5k GPU-hours |
| 4,000 tokens/s | 32 s | 129 s | 25% | 36 h, 9.2k GPU-hours |

The break-even is clean: generation takes longer than training when decode utilisation falls below 2N ÷ (10N/0.4) = 8% of peak, about 1,300 tokens/s per GPU for this model. The steady-state table also flatters generation. A synchronous step waits for its longest responses (up to the 8k limit when the average is 4k), so GPUs idle while the tail finishes, which in practice makes generation the larger share and is the main argument for asynchronous RL (19.10).

**Steps, learning rate and off-policyness.** Full-parameter RL for reasoning uses small, constant learning rates, around 1 × 10⁻⁶ (DeepSeekMath and DAPO both used this value), for on the order of a thousand steps. A thousand steps of 512 prompts is 512k prompt visits, so a filtered pool of about 83k prompts (a third of 250k) is seen about six times, which is why pass rates must be re-estimated. Each rollout of 8,192 responses is usually split into several mini-batches for gradient steps (DAPO used 16 updates per rollout); the last mini-batch is then 15 optimiser steps away from the policy that generated it, and the clipped ratio is what keeps that safe.

**Loss settings that most recent recipes converge on.**

- Token-level averaging over the batch (no per-response length normalisation).
- Clip-higher (0.8 to 1.28).
- No KL penalty or a very small one.
- Truncated samples masked from the loss.
- Dynamic sampling so each batch contains only groups with mixed rewards.
- An importance-sampling correction for the engine–trainer gap (truncated IS) and an fp32 output head.

Change one at a time against a baseline, at a fixed generation budget.

**Dashboard.**

| Signal | Healthy | Warning sign |
|---|---|---|
| Training reward | Rising steadily | Jumps suddenly: check for a verifier exploit |
| Response length | Rising slowly on hard prompts | Rising everywhere, including easy prompts: overthinking or a length bias |
| Policy entropy | Falling slowly | Collapsing in the first few hundred steps |
| Share of zero-signal groups | Stable | Rising: the pool has become too easy or too hard |
| Held-out pass@1 and pass@64 | Both rising | pass@1 up, pass@64 down: sharpening only |

**Reward-hacking audit.** Every few hundred steps, read 20 high-reward samples from the latest checkpoint. Exploits of the checker show up in the text long before they show up in the metrics:

- answers that print the expected output without computing it;
- code that special-cases the tests;
- maths answers that list several values hoping the matcher picks one.

## 19.12 Predicting RL at scale

Pretraining has scaling laws (Chapter 13); RL recipes were compared at one budget, which rewards whichever recipe climbs fastest early. Khatri et al. (2025) ran more than 400,000 GPU-hours of ablations and found that reward against RL compute C follows a **sigmoid**:

R(C) = R₀ + (A − R₀) / (1 + (C_mid / C)^B)

where A is the asymptote (the best reward the recipe can ever reach), B how steeply it gets there and C_mid the compute at the halfway point. Fitting the curve on the first part of a run predicted the rest of a 100,000 GPU-hour run. Two findings matter for practice:

- **Most details change efficiency, not the ceiling.** Loss aggregation, advantage normalisation, curriculum and the choice of off-policy algorithm mostly moved B and C_mid; a few choices (the loss type, fp32 at the output head, the model's size) moved A.
- **Fast early is not best late.** A recipe that wins at a small budget can plateau lower. Compare recipes by fitted asymptote and efficiency, not by a single point.

The resulting recipe (ScaleRL) combines several of the choices in this chapter: asynchronous off-policy generation, CISPO-style loss, prompt-level loss averaging, batch-level advantage normalisation, fp32 logits, zero-variance filtering and dropping prompts the policy already solves. Treat it, like any recipe, as a strong baseline to reproduce rather than a law.

**Prompt-level against token-level averaging.** The two are less different than they sound. Token-level averaging (19.11) divides the sum over every token in the batch by the batch's total token count, so prompts with long responses get more total weight. Prompt-level averaging divides each prompt's summed loss by the total tokens of *its own group* of G responses, then averages over prompts: within a group, every token still has the same weight (so a long wrong answer is not penalised less per token, the bias token-level averaging fixes), but every prompt contributes equally, however long its responses. Both remove the per-response 1/|o_i| of the original GRPO; they differ only in how prompts are weighted against each other. ScaleRL found the choice mattered mainly for efficiency, which is consistent with the first finding above.

## Pitfalls

- Trusting a verifier you have not tried to break.
- Reporting only pass@1, hiding a loss of diversity.
- Comparing GRPO variants with different loss normalisations, prompt sets or length limits.
- Training on prompts with wrong reference answers.
- Ignoring zero-signal groups and wasting most of the compute.
- Using majority voting where the model has a systematic wrong belief.
- Letting response length grow without checking whether it is reasoning or padding.
- Treating rollouts from a separate inference engine as exactly on-policy without measuring the engine–trainer gap.
- Choosing between RL recipes from a single short run: fit the curve and compare asymptotes.
- Describing GRPO as "PPO without a critic" without being able to write its normalisations, which are where the published fixes live.
- Claiming RL taught new reasoning from pass@1 alone, or from problems the base model could already solve at large k.
- Evaluating a reasoning model greedily, or at a different temperature from its baseline.

## Interview questions

**1. How was DeepSeek-R1 trained?** Strong base model; R1-Zero: RL with GRPO and rule-based correctness and format rewards, no SFT; R1: a small cold-start SFT on readable long traces, reasoning RL, then rejection-sampled SFT and a general RL stage; then distillation into small models. Each stage fixes the previous one's problem: the cold start fixes R1-Zero's unreadable, language-mixing traces (helped by a language-consistency reward during RL); the rejection-sampled SFT (about 600k reasoning plus 200k general samples) restores general abilities that reasoning RL ignored; the final RL stage adds helpfulness and harmlessness rewards. The 800k samples also trained the distilled models, which beat RL run directly on models of the same size. Follow-up: "what did R1-Zero prove?" That outcome rewards alone, at scale, elicit long reasoning and self-verification from a strong base model.

**2. Explain GRPO and why it has no value model.** Sample a group per prompt; the advantage is the reward relative to the group mean (often divided by its std); PPO-style clipped updates. The group mean is the baseline, so no learned critic. For rewards [1, 0, 0, 1, 1, 0, 0, 0] the advantages are +1.21 and −0.72. Dropping the critic saves a model the size of the policy and avoids its hardest job, predicting eventual success from a half-written chain of thought, at the price of G samples per prompt. When not to: if rewards are dense per step or episodes are very long and multi-turn, a learned value can still pay. Follow-up: "is the group mean unbiased?" It includes the sample itself, which shrinks the gradient by (G − 1)/G; RLOO's leave-one-out mean does not.

**3. What is the length bias in the original GRPO loss?** Per-response averaging gives each token weight A/|o|, so long wrong answers are penalised less per token, encouraging length growth on failures. Fix: token-level averaging or a constant normaliser. In the chapter's group, a correct 200-token answer gets +6.04e-3 per token while wrong answers of 1,500 and 2,000 tokens get −0.48e-3 and −0.36e-3; with token-level averaging every correct token gets +0.18e-3 and every wrong one −0.11e-3. In total weight, the 2,000-token wrong answer goes from −0.091 (the same as a 400-token one) to −0.215. Follow-up: "why does it matter for 'emergent' length growth?" Part of the length increase reported in early R1-style runs is this artefact, so measure length growth under both normalisations before calling it reasoning.

**4. What does DAPO change?** Clip-higher (asymmetric clipping), dynamic sampling (drop groups with no reward variance), token-level loss, overlong reward shaping, no KL penalty. Each targets a measured failure: entropy collapse (the upper clip at 1.28 lets a token at p = 0.01 reach 0.0128 instead of 0.012), wasted compute on zero-signal groups (at G = 8 a prompt at a 5% pass rate gives nothing 66% of the time), the length bias above, and noisy rewards on truncated responses. Removing the KL assumes the policy should move far from the reference, which holds for long-reasoning RL but not for preference tuning. Follow-up: "which change matters most?" Report ablations at a fixed generation budget; published gains are often recipe-specific.

**5. Why do groups where all answers are right (or wrong) matter?** Zero advantage, zero gradient. At extreme pass rates most groups give no signal; filter by difficulty or resample. The probability of a dead group is p^G + (1 − p)^G: 85% at a 2% pass rate with G = 8, 66% at 5%, under 1% at 50%. Doubling G to 16 cuts the 5% case to 44%, at twice the generation, so dynamic sampling and a 10–90% difficulty filter are cheaper than ever-larger groups. Follow-up: "what happens over training?" Prompts drift towards 100% as the policy improves, so the share of dead groups rises unless pass rates are re-estimated and the pool refilled.

**6. Does RL teach models new reasoning?** Evidence is mixed: RL raises pass@1 and often lowers pass@k at large k, suggesting sharpening of existing solutions; longer, more exploratory RL may extend the frontier; distillation from stronger models adds capability. Measure pass@k curves. The chapter's toy makes the mechanism concrete: pass@1 rises from 0.337 to 0.465 but pass@64 falls from 0.910 to 0.763, and on the hard quarter needing a rare strategy from 0.402 to 0.032, because the rare strategy is squeezed out. Yue et al. found the same crossing in real models; ProRL's long, reset-stabilised runs reported gains at large k where the base model failed. The decisive experiment: held-out problems the base model fails at k = 1,024, same temperature, strict checking. My view: short narrow RL sharpens; long diverse RL with exploration kept alive can extend, within limits set by pretraining.

**7. PRM versus ORM?** ORMs (or verifiers) judge final answers; PRMs judge each step. PRMs give denser signal and better reranking and search but are expensive to label and easy to hack as RL rewards. Lightman et al. needed 800k human step labels (PRM800K); Math-Shepherd replaced them with Monte Carlo estimates (the fraction of continuations from a step that reach the right answer), which costs many rollouts per step and inherits their noise. As an RL reward, a PRM is a learned model, so the policy learns steps that look good to it; R1-style training avoided PRMs for that reason. Where they earn their cost is at inference: reranking and step-level search. Follow-up: "how would you get process signal without a PRM?" Token- or step-level advantages from rollouts, or on-policy distillation from a teacher.

**8. When does self-consistency fail?** When a wrong answer is more probable than the right one: voting then amplifies the error. It also needs comparable final answers. With 30% single-sample accuracy and wrong answers spread over five options, majority voting over 25 samples reaches 70%; if one wrong answer takes the other 70%, the vote is right only 2% of the time. The cheap check is to sample 20–40 answers on a few hundred dev problems and look at answer histograms before paying for voting. For free-form outputs (proofs, code), cluster answers by an equivalence check or use a verifier instead.

**9. How would you reduce overthinking on easy questions?** Length penalties relative to group or difficulty, adaptive budgets, training on mixed difficulty with rewards for efficiency, routing easy queries to a non-thinking mode. A length penalty relative to the group (shorter correct answers get a bonus over longer correct ones) is safer than an absolute one, which can teach the model to give up on hard problems. Hybrid models make routing explicit: Qwen3-style thinking and non-thinking modes, or an effort setting, chosen by a classifier or the user. The number to quote: 4,000 thinking tokens before a 200-token answer is about 20× the output cost of answering directly. Follow-up: "how do you know you have not hurt hard problems?" Track accuracy by difficulty bucket, not on average.

**10. How do you make a 7B model reason well cheaply?** Distil verified long reasoning traces from a stronger model (SFT), then optionally a short RL stage on verifiable tasks at appropriate difficulty. Scale: s1 and LIMO showed about 1,000 carefully chosen traces already elicit much of the behaviour from a strong base; DeepSeek's distilled models used about 800k, and on small models that beat running RL directly. Cost: generating 100k traces of 8k tokens is 0.8B teacher tokens, small next to an RL run that generates over 30M tokens per step for a thousand steps. RL on top still pays when you have a good verifier and the student's pass rates on your target tasks sit in the 10–90% band; on-policy distillation is the middle option, dense like SFT and on-policy like RL, reported at about a tenth of RL's GPU hours. Follow-up: "what goes wrong with distillation?" The student copies the teacher's verbosity and errors unless traces are verified and filtered.

**11. Why is RL for reasoning expensive, and how do systems handle it?** Generation of long rollouts dominates; separate inference engines, asynchronous pipelines with bounded staleness, length-aware batching, careful handling of probability mismatches between engine and trainer. The arithmetic: a 30B model training on 32.8M tokens per step needs 10N FLOPs per token (update, reference, old log-probs), 97 s at 40% MFU on 256 GPUs; generation needs only 2N but decodes at low utilisation, and it dominates whenever utilisation is below 8% of peak (about 1,300 tokens/s per GPU here), plus the synchronous step waits for the longest responses. Hence asynchronous RL, which keeps both sides busy at the price of slightly stale samples. Follow-up: "how stale is acceptable?" Bound it to one or a few policy versions and correct with clipped or truncated importance weights.

**12. What is the training–inference mismatch in RL, and how do you handle it?** The rollout engine and the trainer give slightly different probabilities for the same tokens, so the data is off-policy. Recompute log-probabilities in the trainer, apply truncated importance weights, keep the output head in fp32, monitor the per-token gap, or use batch-invariant kernels. Why it bites: per-token differences compound over a sequence; at a log-ratio standard deviation of only 0.03, a 4,000-token sequence's ratio has variance 35.6 (Chapter 18.4). Truncated IS weights each token by min(π_trainer/π_engine, C): bounded variance for a little bias, which also makes 8-bit rollouts usable. Follow-up: "what is the early warning?" A rising engine–trainer KL often precedes a collapse.

**13. GRPO, GSPO and CISPO: what differs?** Same group-relative advantage. GRPO clips per-token ratios; GSPO uses one length-normalised sequence ratio and clips whole responses (more stable for MoE); CISPO clips the importance weight but keeps the gradient of every token, so rare high-ratio tokens still learn. The underlying tension: the reward belongs to the sequence but the ratio to the token. GSPO's sequence ratio is exp(mean per-token log-ratio), which keeps it near 1 even for long responses, unlike the raw product. CISPO's objective is stop-gradient(clipped weight) × Â × ∇log π, so a "wait" token whose ratio jumped above 1 + ε still gets a bounded gradient instead of none. Follow-up: "when would you pick GSPO?" MoE RL, where routing changes make per-token ratios noisy.

**14. How would you do RL where answers can't be checked?** Per-prompt rubrics scored by an LLM judge (weighted share of criteria met), generative reward models, or pairwise preferences; audit high-reward samples for judge exploitation and length bias, and validate the judge against expert labels. A rubric turns one vague judgement into several checkable ones ("mentions the interaction", "does not invent a dosage"), and Gunjal et al. found rubric rewards beat a single Likert judge score. The risks are the judge's: verbosity, rubric items recited rather than satisfied, and shared blind spots. Measure judge–expert agreement per criterion on a few hundred items before training, and keep a held-out human evaluation. Follow-up: "how is this different from RLHF?" The criteria are explicit per prompt, so the reward is narrower and harder to game than a general preference model.

**15. What is on-policy distillation and when would you use it?** Sample from the student, score each token by the teacher (reverse KL as a dense reward). Use it to make a small model match a strong teacher cheaply, or to recover skills lost after domain fine-tuning; it needs teacher log-probabilities on a shared vocabulary. It combines RL's on-policy sampling (the student learns in states it actually reaches, fixing SFT's exposure bias) with a dense per-token signal instead of one bit per episode. The Qwen3 report found it beat RL on the same student (74.4% against 67.6% on AIME'24) at about a tenth of the GPU hours. When not to: no access to teacher log-probabilities, a different tokenizer, or a target beyond the teacher's own ability.

**16. How do you compare two RL recipes fairly?** Same model, prompts, verifier and generation budget; several runs if affordable; fit a saturating curve of reward against compute and compare asymptote and efficiency, not one checkpoint; report pass@k curves and held-out evaluations. The reason: Khatri et al. found that most recipe details change how fast reward rises, not where it ends, so a recipe that wins early can plateau lower. Seeds matter too: RL runs are noisy, and a difference smaller than the spread of two seeds is not a result. Follow-up: "you can afford one run each?" Then compare at several compute points along the curve and say the conclusion is provisional.

**17. Write the full GRPO objective and say where the KL sits.** J = E[(1/G) Σ_i (1/|o_i|) Σ_t (min(ρ_(i,t)Â_i, clip(ρ_(i,t), 1 − ε, 1 + ε)Â_i) − β·D_(i,t))], where ρ_(i,t) = π_θ(o_(i,t) | q, o_(i,<t)) / π_old(·), Â_i = (r_i − mean r)/std r is shared by every token of output i, and D_(i,t) = π_ref/π_θ − log(π_ref/π_θ) − 1 is the k3 KL estimate. The KL sits in the loss, as a per-token regulariser, not in the reward as in InstructGPT-style PPO, so the advantage contains only the verifier's signal. Each term is a fix site: 1/|o_i| is the length bias (DAPO and Dr. GRPO replace it), the std is the difficulty weighting (Dr. GRPO drops it), the per-token ρ is what GSPO and CISPO change, and β is often set to zero for long reasoning. Follow-up: "compute one token": ρ = 1.3, Â = +1.21 gives min(1.573, 1.452) = 1.452.

**18. What is entropy collapse and why does clip-higher help?** Policy entropy, the mean over generated positions of the next-token distribution's entropy, falls fast under RL as the policy commits to strategies that already work; once it is near zero the policy stops exploring and pass@k at large k falls. The clip bounds the probability ratio, a multiplicative change, so a token at p = 0.01 can gain at most 0.002 per update with ε = 0.2, while one at 0.5 can gain 0.1: rare, useful tokens grow slowest. Raising only the upper bound to 1.28 lets them grow faster without letting probabilities fall faster. Other tools: temperature, entropy bonuses, removing the KL. Follow-up: "what entropy is healthy?" Slowly falling; a collapse in the first few hundred steps is the warning sign.

**19. Estimate the cost of one RLVR step for a 30B model with 512 prompts × 16 samples × 4k tokens.** Tokens: 32.8M. Training side: 10N FLOPs per token (6N update, 2N reference, 2N old-policy log-probs) = 9.8e18 FLOPs, 97 s on 256 GPUs at 40% MFU. Generation: 2N per token but memory-bound; at 2,000 tokens/s per GPU it takes 64 s, at 1,000 it takes 128 s. So a step is about 2.5–4 minutes and 1,000 steps is roughly 10–16k GPU-hours on H100-class hardware, before tail effects. The judgement: generation dominates when decode utilisation is below 8% of peak and because synchronous steps wait for the longest responses, so the first optimisations are decode throughput (TP for KV room, fp8 rollouts with truncated IS) and asynchronous generation. Follow-up: "what if you use G = 8?" Half the generation, but more dead groups.

## Practice

1. Implement GRPO with token-level loss for a small model (0.5–1.5B) on grade-school maths with an exact-match verifier; log reward, response length and entropy.

<details><summary>Hint and expected result</summary>

Unit-test the loss before training: on the group of 19.4 (rewards [1, 0, 0, 1, 1, 0, 0, 0], lengths [200, 900, 1,500, 300, 250, 400, 1,200, 2,000]) the per-token weights must be +0.18e-3 and −0.11e-3 under token-level averaging. Use 8–16 samples per prompt and a learning rate near 1e-6. Expect training reward to rise over a few hundred steps on a model that starts with a non-trivial pass rate, and entropy to fall; a reward jump within a few steps usually means a verifier exploit or an answer-format change, so read samples.

</details>

2. Compare sequence-mean and token-mean normalisation on the same run; plot length over training.

<details><summary>Hint and expected result</summary>

Hold everything else fixed, including the seed and the prompt order. Expect response length to grow faster under sequence-mean normalisation, especially on prompts the model fails, and to stay flatter under token-mean; plot length separately for correct and wrong responses to see where the growth comes from. Run two seeds of each: length curves are noisy and a single pair can mislead.

</details>

3. Write a maths answer verifier, then try to break it with adversarial outputs; fix what you find.

<details><summary>Hint and expected result</summary>

Try: several boxed answers, an answer followed by "or", units and formatting variants (1/2, 0.5, \frac{1}{2}), very long outputs that contain the answer somewhere, answers in the reasoning but not the final line, and LaTeX that crashes the parser. Expect a naive regex-and-compare verifier to fail on at least a few of these; a good one parses only the final answer, normalises symbolically and treats parse failures as wrong.

</details>

4. **Research exercise.** For a base model and its RL-trained version (open pairs exist), estimate pass@k for k = 1 to 256 on 200 maths problems with the unbiased estimator (Chapter 13). Plot both curves with bootstrap confidence intervals. Where do they cross, and on which problems does the base model win?

<details><summary>Hint and expected result</summary>

Use n = 256 samples per problem at the same temperature for both models and bootstrap over problems. Expected outcome, following Yue et al.: the RL model wins clearly at k = 1, the gap narrows as k grows, and the curves often cross somewhere between k = 16 and k = 256. The problems where the base model wins at large k usually need an approach the RL model has stopped sampling, as in the chapter's toy. Then run the decisive experiment of 19.6 on problems the base model never solves.

</details>

5. Implement self-consistency and best-of-n with a small reward model; compare accuracy against n.

<details><summary>Hint and expected result</summary>

First reproduce the chapter's simulation: with 30% single-sample accuracy and wrong answers over five options, majority voting reaches about 70% at n = 25; with one dominant wrong answer it falls to about 2%. On real data expect self-consistency to rise and saturate within 16–64 samples; best-of-n with a reward model rises faster at small n and can turn down at large n if the reward model is exploitable (Chapter 18.5).

</details>

6. Compute the cost table of 19.11 for your own setting: model size, GPUs, measured decode throughput, tokens per step.

<details><summary>Hint and expected result</summary>

Use 10N FLOPs per training token and your measured generation tokens/s. For the chapter's 30B setting at 2,000 tokens/s per GPU on 256 GPUs: 64 s of generation, 97 s of training, 161 s per step, 11.5k GPU-hours per 1,000 steps. Then measure the real step and explain the gap (tail latency, weight-sync time, idle GPUs while waiting for the slowest responses).

</details>

## Read next

- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025).
- Shao et al., "DeepSeekMath" (2024), for GRPO.
- Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO, 2025); Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
- Kimi Team, "Kimi k1.5: Scaling Reinforcement Learning with LLMs" (2025).
- Lightman et al., "Let's Verify Step by Step" (2023); Wang et al., "Math-Shepherd" (2023).
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (2022); Muennighoff et al., "s1: Simple test-time scaling" (2025).
- Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (2025); Liu et al., "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models" (2025); Chu et al., "SFT Memorizes, RL Generalizes" (2025).
- Qwen Team, "Qwen3 Technical Report" (2025), for thinking-mode fusion and on-policy distillation.
- Zheng et al., "Group Sequence Policy Optimization" (2025); MiniMax, "MiniMax-M1" (2025), for CISPO.
- Khatri et al., "The Art of Scaling Reinforcement Learning Compute for LLMs" (2025).
- Yao et al., "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (2025); He et al., "Defeating Nondeterminism in LLM Inference" (2025).
- Gunjal et al., "Rubrics as Rewards" (2025); Lu et al., "On-Policy Distillation" (2025).
- Links to these papers are in the Key concepts section at the top of this chapter.

## Done when

- Without notes, write the full GRPO objective with the ratio, clip, group advantage and k3 KL term in 5 minutes, and say where each 2025 fix changes it.
- Compute the group advantages for rewards [1, 0, 0, 1, 1, 0, 0, 0] (+1.21, −0.72) and the per-token weights for lengths [200, 900, 1,500, 300, 250, 400, 1,200, 2,000] under sequence-mean (+6.04e-3 for the 200-token answer, −0.36e-3 for the 2,000-token one) and token-mean (+0.18e-3, −0.11e-3).
- Compute the probability of a zero-signal group at a 5% pass rate for G = 8 and 16 (66%, 44%).
- Explain why clip-higher helps rare tokens with the numbers 0.01 → 0.012 against 0.0128.
- Describe the four R1 stages and what each fixes.
- Argue both sides of whether RL creates new reasoning with the toy result (pass@64 0.910 → 0.763) and the ProRL and Chu et al. evidence, and design the decisive experiment.
- Say when self-consistency fails, with the 70% and 2% examples.
- Estimate the cost of an RLVR step for a 30B model at 32.8M tokens: 97 s of training-side compute on 256 GPUs at 40% MFU plus generation at your decode throughput; state when generation dominates (below 8% decode utilisation).
- Explain the engine–trainer mismatch and its fixes, compare GRPO, GSPO and CISPO, design a rubric reward, and compare two RL recipes by fitted curves rather than a single checkpoint.

---

[← 18 · Post-training I: SFT, reward models and preference optimisation](18-post-training-sft-and-preferences.md) · [Contents](../README.md#contents) · [20 · Fine-tuning and parameter-efficient methods →](20-fine-tuning-and-lora.md)
