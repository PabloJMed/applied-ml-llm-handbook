# 19 · Post-training II: reinforcement learning and reasoning models

Part V · Post-training. Assumes Chapter 18 (PPO, advantages, clipping, KL, reward over-optimisation) and Chapter 13 (test-time compute, pass@k). Chapter 33 extends the same methods to multi-step agents; Chapter 27 covers reward hacking as an alignment problem.

## Why it matters

From late 2024 the most important advances in LLMs came from reinforcement learning on tasks whose answers can be checked: maths with known answers, code with tests, puzzles and tool tasks with verifiable end states. Models trained this way think at length before answering, check and revise their work, and scale their accuracy with the inference compute they are given. DeepSeek-R1's open report (2025) showed the recipe could be reproduced; since then, RL has become a large and growing share of post-training compute. For a research scientist, this is the most active area of the field: the algorithms (GRPO and its fixes), reward design and verifiers, curricula, the debate about whether RL teaches new reasoning or sharpens what pretraining already gave, and how to spend inference compute well.

Numbers in this chapter come from `code/ch19_check.py` (NumPy, under a second).

## Key concepts

- **RL with verifiable rewards (RLVR)**: reward from a programmatic check of the final answer. [Tülu 3 (RLVR)](https://arxiv.org/abs/2411.15124), [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
- **GRPO**: policy optimisation with group-relative advantages and no value model. [DeepSeekMath](https://arxiv.org/abs/2402.03300).
- **Fixes to GRPO**: [Dr. GRPO (Understanding R1-Zero-like training)](https://arxiv.org/abs/2503.20783), [DAPO](https://arxiv.org/abs/2503.14476).
- **Process reward models (PRMs)**: score each reasoning step, not only the answer. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), [Math-Shepherd](https://arxiv.org/abs/2312.08935).
- **Test-time scaling**: self-consistency, best-of-n, search, longer thinking. [Self-consistency](https://arxiv.org/abs/2203.11171), [Snell et al.](https://arxiv.org/abs/2408.03314), [s1 (budget forcing)](https://arxiv.org/abs/2501.19393).
- **Does RL add new capability?** [Yue et al.](https://arxiv.org/abs/2504.13837).
- **Scaling RL in practice**: [Kimi k1.5](https://arxiv.org/abs/2501.12599).

## 19.1 What changed

Earlier attempts to teach reasoning through supervised chains of thought or learned process rewards helped but did not scale cleanly. The recipe that worked is simple:

1. Start from a strong base model (pretraining and mid-training on maths and code matter a lot).
2. Optionally, a small SFT "cold start" on long, well-formatted reasoning traces.
3. Run RL on many prompts with checkable answers, rewarding correctness (and format), with a policy-gradient method such as GRPO.
4. Distil the resulting behaviour into smaller models, and mix reasoning data with general post-training (Chapter 18).

DeepSeek-R1-Zero went straight from base model to RL with no SFT and still learned long reasoning, self-verification and backtracking (the "aha moment" of the report), though with readability problems that the cold-start SFT fixed in R1. Response length grew steadily during training: the model learned to use more tokens because more thinking earned more reward.

## 19.2 Verifiers and reward design

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

## 19.3 GRPO in detail

For each prompt, sample a group of G responses (8 to 64), score each with the verifier, and use the score relative to the group as the advantage of every token in that response:

A_i = (r_i − mean(r)) / std(r)

No value model is needed: the group is the baseline. The objective is PPO's clipped surrogate (Chapter 18) applied per token with this advantage, often with a KL penalty to the reference model.

For rewards [1, 0, 0, 1, 1, 0, 0, 0], the advantages are +1.21 for correct and −0.72 for wrong responses.

**Why groups work.** A value model for long reasoning must predict, from a partial chain of thought, the chance of eventual success: hard to learn and memory-hungry. A group of samples estimates the prompt's baseline directly, at the price of more generation.

## 19.4 Known biases and the fixes

Careful analysis in 2025 found that the original GRPO objective has biases, and that several engineering choices matter as much as the algorithm.

**Length normalisation bias.** Original GRPO averages the loss over the tokens of each response, then over responses. Each token's weight is A_i / |o_i|. For a correct 200-token answer that is +6.04e-3 per token; for wrong answers of 1,500 and 2,000 tokens it is −0.48e-3 and −0.36e-3. Longer wrong answers are penalised less per token, so the model drifts towards longer outputs when it is wrong. Normalising by a constant (Dr. GRPO) or averaging over all tokens in the batch (DAPO's token-level loss) gives every token the same weight: +0.18e-3 for correct tokens and −0.11e-3 for wrong ones, whatever the length.

**Standard-deviation normalisation.** Dividing by the group's standard deviation up-weights prompts that are nearly always solved or nearly always failed (small std), which are not necessarily the most informative. Dr. GRPO drops the division; others keep it. The choice changes which prompts dominate learning.

**Zero-signal groups.** If every response in a group gets the same reward, all advantages are zero and the prompt contributes nothing. With 8 samples, a prompt with a 2% or 98% pass rate gives no signal 85% of the time; one at 50% almost never. **Dynamic sampling** (DAPO) keeps sampling until the batch is filled with groups that have mixed results.

**Entropy collapse and clip-higher.** RL quickly reduces the policy's entropy: it commits to the strategies that already work and stops exploring. Symmetric clipping at 1 ± 0.2 lets a token with probability 0.01 rise to only 0.012 per update, while confident tokens can be reinforced freely. DAPO's **clip-higher** raises the upper bound (to 1.28), which lets rare, useful tokens grow faster and keeps entropy up. Other tools: entropy bonuses, a higher sampling temperature, and removing the KL penalty (DAPO removes it, since the policy is supposed to move far from the reference in long-reasoning RL).

**Overlong responses.** Responses cut at the length limit get no reward though their reasoning may be fine, which adds noise. DAPO masks truncated samples from the loss or applies a soft penalty as responses approach the limit.

Other algorithms in the same family: **RLOO** (leave-one-out baseline within the group), **REINFORCE++** and various "critic-free" PPO variants. Treat claims that one variant is better with care: reported differences often come from the implementation details above, from learning rates, or from the prompt set.

## 19.5 Data and curriculum

- **Difficulty**: keep prompts whose pass rate under the current policy is neither 0 nor 1 (often 10–90%). Re-estimate as the policy improves; prompts that become easy stop teaching.
- **Coverage**: maths of several levels, code in several languages, logic puzzles, STEM questions with short answers, instruction constraints. Narrow prompt sets give narrow gains.
- **Verifiability first**: drop prompts whose reference answers are wrong or ambiguous (a surprising fraction in scraped maths sets); these teach the model to fit noise.
- **Curricula**: increase difficulty or maximum response length in stages; long-context reasoning is expensive, so many runs grow the length limit over training.

## 19.6 Does RL teach new reasoning?

Yue et al. (2025) compared base and RL-trained models with pass@k for large k. RL models win clearly at k = 1, but base models often catch up or win at k = 256: RL mostly **sharpens** the distribution towards solutions the base model could already sample, and can **narrow** it, losing rare solutions. Others argue that longer RL, larger models or more diverse prompts do extend the frontier, and that distillation from a stronger model genuinely adds new capability to a small one. The debate is about evidence, so you should be able to produce some.

The check script builds a toy version: 200 problems, six solution strategies; one common strategy works on 75% of problems, and the remaining 25% need a rare strategy the base policy picks only 1% of the time. GRPO-style updates on a shared policy:

| | pass@1 | pass@64 | Entropy | pass@64 on the hard 25% |
|---|---|---|---|---|
| Base policy | 0.337 | 0.910 | 1.03 | 0.402 |
| After RL | 0.465 | 0.763 | 0.05 | 0.032 |

pass@1 rises; pass@64 falls, because the rare strategy, reinforced too seldom to compete, is squeezed out. This is entropy collapse in miniature. The lesson for practice: **report pass@k curves, not only pass@1**, monitor entropy, and keep exploration alive (clip-higher, temperature, diverse prompts) if you want RL to find new solutions rather than only sharpen old ones.

## 19.7 Process supervision

An outcome reward model (ORM), or a verifier, judges the final answer. A **process reward model (PRM)** scores each step of the reasoning. Lightman et al. (2023) trained a PRM on 800k human step labels (PRM800K) and found that, for selecting the best of many solutions, process supervision beat outcome supervision on competition maths. Math-Shepherd labelled steps automatically: a step's label is the fraction of continuations from that step that reach the right answer (a Monte Carlo estimate of its value).

PRMs are used for: reranking samples, guiding search (beam search over steps, tree search), and dense rewards in RL. Their problems: human step labels are expensive; automatic labels are noisy; and PRMs used as RL rewards get hacked (the policy produces steps that look good to the PRM). R1-style training used outcome rewards only, and most RLVR pipelines still do; PRMs are more common at inference.

## 19.8 Spending inference compute

| Strategy | How | Needs |
|---|---|---|
| Self-consistency | Sample n reasoning paths; majority vote on the final answer | Answers that can be compared |
| Best-of-n | Sample n; pick the highest-scoring by a verifier or reward model | A good verifier |
| Search with a PRM | Beam or tree search over reasoning steps | A PRM; more engineering |
| Sequential revision | Answer, critique, revise | A model that can find its own errors |
| Longer thinking | Train the model to think longer; control the budget | RL-trained reasoning model |

**Self-consistency** works when the correct answer is the most common one. With a 30% single-sample accuracy and wrong answers spread over five options, majority voting over 25 samples reaches 70%; at 50% accuracy it reaches 99%. But if one wrong answer is more popular than the right one, voting makes things worse: with two wrong options, 30% accuracy and 25 votes, accuracy collapses to 2%. Voting amplifies whatever the model believes most.

**Budget forcing** (s1, 2025) controls the thinking length at inference: stop thinking early by forcing the end-of-thinking token, or extend it by suppressing that token and appending "Wait". With only 1,000 curated SFT examples, accuracy rose with the forced budget. Reasoning models also **overthink**: they spend thousands of tokens on easy questions. Length penalties and adaptive budgets are an active area.

## 19.9 Distilling reasoning

The cheapest way to give a small model reasoning ability is SFT on reasoning traces from a stronger model. DeepSeek released distilled models trained on about 800k samples from R1 that outperformed the same small models trained with RL directly. s1 and LIMO showed that around a thousand carefully chosen, hard, verified traces already elicit much of the behaviour from a strong base model. The emerging rule: **distil into small models, run RL on large ones**, then distil again. Quality filters on traces matter: verify final answers, remove repetitive or incoherent thinking, prefer diverse problems.

## 19.10 Systems for RL

RL for reasoning is dominated by **generation**: sampling long responses for many prompts takes most of the time. Practical consequences:

- Rollouts run in a fast inference engine (vLLM, SGLang), training in a training framework; weights are synced after each update.
- **Asynchronous RL** overlaps generation and training, which makes samples slightly off-policy; importance weights or bounded staleness keep this in check.
- **Training–inference mismatch**: the inference engine and the trainer compute slightly different token probabilities (different kernels, precision). The PPO ratio then mixes real policy change with numerical noise, which can destabilise long runs. Fixes include recomputing log-probabilities in the trainer and importance-sampling corrections.
- Long responses create imbalanced batches; pack and sort by length.
- Open frameworks: verl, OpenRLHF, TRL, and others (Chapter 32).

## Pitfalls

- Trusting a verifier you have not tried to break.
- Reporting only pass@1, hiding a loss of diversity.
- Comparing GRPO variants with different loss normalisations, prompt sets or length limits.
- Training on prompts with wrong reference answers.
- Ignoring zero-signal groups and wasting most of the compute.
- Using majority voting where the model has a systematic wrong belief.
- Letting response length grow without checking whether it is reasoning or padding.

## Interview questions

**1. How was DeepSeek-R1 trained?** Strong base model; R1-Zero: RL with GRPO and rule-based correctness and format rewards, no SFT; R1: a small cold-start SFT on readable long traces, reasoning RL, then rejection-sampled SFT and a general RL stage; then distillation into small models.

**2. Explain GRPO and why it has no value model.** Sample a group per prompt; the advantage is the reward relative to the group mean (often divided by its std); PPO-style clipped updates. The group mean is the baseline, so no learned critic.

**3. What is the length bias in the original GRPO loss?** Per-response averaging gives each token weight A/|o|, so long wrong answers are penalised less per token, encouraging length growth on failures. Fix: token-level averaging or a constant normaliser.

**4. What does DAPO change?** Clip-higher (asymmetric clipping), dynamic sampling (drop groups with no reward variance), token-level loss, overlong reward shaping, no KL penalty.

**5. Why do groups where all answers are right (or wrong) matter?** Zero advantage, zero gradient. At extreme pass rates most groups give no signal; filter by difficulty or resample.

**6. Does RL teach models new reasoning?** Evidence is mixed: RL raises pass@1 and often lowers pass@k at large k, suggesting sharpening of existing solutions; longer, more exploratory RL may extend the frontier; distillation from stronger models adds capability. Measure pass@k curves.

**7. PRM versus ORM?** ORMs (or verifiers) judge final answers; PRMs judge each step. PRMs give denser signal and better reranking and search but are expensive to label and easy to hack as RL rewards.

**8. When does self-consistency fail?** When a wrong answer is more probable than the right one: voting then amplifies the error. It also needs comparable final answers.

**9. How would you reduce overthinking on easy questions?** Length penalties relative to group or difficulty, adaptive budgets, training on mixed difficulty with rewards for efficiency, routing easy queries to a non-thinking mode.

**10. How do you make a 7B model reason well cheaply?** Distil verified long reasoning traces from a stronger model (SFT), then optionally a short RL stage on verifiable tasks at appropriate difficulty.

**11. Why is RL for reasoning expensive, and how do systems handle it?** Generation of long rollouts dominates; separate inference engines, asynchronous pipelines with bounded staleness, length-aware batching, careful handling of probability mismatches between engine and trainer.

## Practice

1. Implement GRPO with token-level loss for a small model (0.5–1.5B) on grade-school maths with an exact-match verifier; log reward, response length and entropy.
2. Compare sequence-mean and token-mean normalisation on the same run; plot length over training.
3. Write a maths answer verifier, then try to break it with adversarial outputs; fix what you find.
4. **Research exercise.** For a base model and its RL-trained version (open pairs exist), estimate pass@k for k = 1 to 256 on 200 maths problems with the unbiased estimator (Chapter 13). Plot both curves with bootstrap confidence intervals. Where do they cross, and on which problems does the base model win?
5. Implement self-consistency and best-of-n with a small reward model; compare accuracy against n.

## Read next

- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025).
- Shao et al., "DeepSeekMath" (2024), for GRPO.
- Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO, 2025); Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
- Kimi Team, "Kimi k1.5: Scaling Reinforcement Learning with LLMs" (2025).
- Lightman et al., "Let's Verify Step by Step" (2023); Wang et al., "Math-Shepherd" (2023).
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (2022); Muennighoff et al., "s1: Simple test-time scaling" (2025).
- Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (2025).
- Links to these papers are in the Key concepts section at the top of this chapter.

## Done when

You can describe the R1 recipe and what each stage contributes. You can implement GRPO, explain its length and standard-deviation biases and the DAPO and Dr. GRPO fixes. You can design verifiers and a prompt curriculum, and explain how verifiers get gamed.

You can argue both sides of whether RL creates new reasoning, using pass@k curves. You can compare PRMs and ORMs, choose among self-consistency, best-of-n, search and longer thinking, and say when voting fails. You can explain why distillation is the cheap path for small models and what makes RL systems expensive.

---

[← 18 · Post-training I: SFT, reward models and preference optimisation](18-post-training-sft-and-preferences.md) · [Contents](../README.md#contents) · [20 · Fine-tuning and parameter-efficient methods →](20-fine-tuning-and-lora.md)
