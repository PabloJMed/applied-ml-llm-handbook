# 36 · Research interviews

Part X · Research craft and interviews. Builds on Chapter 34 (research craft), Chapter 35 (coding rounds) and Chapter 37 (depth questions). Chapter 38 covers behavioural rounds, including research stories.

## Why it matters

Interviews for research scientist roles in LLMs test different things from applied ML loops. Besides coding and depth questions, you will usually face a **research design** round ("how would you test whether X improves Y?"), a **paper discussion** (yours or someone else's), a **research presentation** on your past work, and sometimes a **take-home research task**. These rounds test judgement: whether you ask the right question, design an experiment that answers it, anticipate confounds, read results sceptically and communicate clearly. This chapter explains what each round looks for and works through example answers.

## Key concepts

- **Research design round**: an open problem, a hypothesis, an experiment plan, risks and decision rules.
- **Paper discussion**: summarise, critique, and propose next experiments for a paper.
- **Research presentation (job talk)**: 30–45 minutes on your work, then questions.
- **Take-home task**: a small research project with a written report, judged on rigour and communication.
- **Signals interviewers look for**: problem framing, experimental rigour, knowledge of the literature, honesty about uncertainty, clear communication. See also [Schulman's guide to ML research](http://joschu.net/blog/opinionated-guide-ml-research.html) for the habits these rounds probe.

## 36.1 The loop

A typical loop for an LLM research scientist (varies by lab and team):

| Round | Length | What it tests |
|---|---|---|
| Recruiter and hiring-manager screens | 30–45 min each | Fit, motivation, areas of interest |
| Coding | 45–60 min, 1–2 rounds | Implementing ML components from scratch (Chapter 35); sometimes general algorithms |
| ML depth | 45–60 min | Fundamentals to research frontier (Chapter 37) |
| Research design | 45–60 min | Turning a vague question into a sound experimental programme |
| Paper discussion or research presentation | 45–60 min | Depth on your work; critical reading |
| Behavioural / collaboration | 30–45 min | Research stories, collaboration, judgement (Chapter 38) |
| Take-home (some teams) | A few hours to a few days | End-to-end research execution and writing |

Before the loop, find out which rounds you will have, what the team works on, and read their recent papers or blog posts. Interviewers often build research-design questions around their own team's problems.

## 36.2 The research design round

You get an open question, such as: "We think training on synthetic maths data improves general reasoning. How would you find out?" There is no single right answer; the interviewer watches how you think.

**A structure that works:**

1. **Clarify the question.** What decision does it inform? What counts as "general reasoning"? Which stage of training (pretraining, mid-training, post-training)? What resources exist?
2. **State hypotheses**, including the boring alternatives: the gain comes from more tokens, from overlap with benchmark formats, from contamination, or from a stylistic change rather than reasoning.
3. **Design the minimal experiment** that distinguishes them: controls, matched compute, metrics.
4. **Plan for noise and scale**: seeds, evaluation precision, at which scales.
5. **Anticipate results and decisions**: what you would conclude and do for each plausible outcome.
6. **Risks and extensions**: what could invalidate the result; the next experiment.

### Worked example: synthetic maths data and reasoning

*Clarify.* Assume we decide whether to include a synthetic maths dataset (say 50B tokens) in the mid-training mixture of our next 8B model. "General reasoning" means held-out reasoning tasks outside maths: logic puzzles, science questions, multi-step word problems in other domains, and code.

*Hypotheses.*
- H1: synthetic maths improves transferable reasoning.
- H0a: gains are confined to maths benchmarks (in-domain only).
- H0b: gains come from extra tokens of any good data, not from maths specifically.
- H0c: gains on benchmarks come from format overlap or contamination of the synthetic data with test items.
- Risk: it hurts other abilities (knowledge, writing) by displacing other data.

*Design.*
- Start from the same pretraining checkpoint; anneal (Chapter 16) with three mixtures at matched tokens: baseline mixture; baseline with 20% replaced by synthetic maths; baseline with 20% replaced by an equally "high-quality" non-maths source (controls for H0b).
- Decontaminate the synthetic data against all evaluations (n-gram and embedding search) and also measure performance on post-cut-off or perturbed problems (H0c).
- Evaluation: in-domain maths; out-of-domain reasoning suites; a regression suite (knowledge, writing, multilingual); per-domain held-out loss.
- Scale: run at two small scales (for example 400M and 1.4B) with 3 seeds of the anneal each; if the effect is consistent in sign and growing or stable, confirm at 8B with one run.

*Noise.* Estimate seed variance from the baseline runs; use paired comparisons on the same items, continuous scores where benchmarks are near chance at small scale (Chapter 23).

*Decisions.* If out-of-domain reasoning improves beyond the non-maths control with intervals excluding zero and no meaningful regressions: include it, at the proportion that maximises the trade-off, and run a proportion sweep. If only maths improves: include only if maths is a product goal, and say so honestly. If the control matches it: the effect is "good data", not maths; prioritise quality filtering instead.

*Extensions.* Which properties of the synthetic data matter (difficulty, verification, diversity)? Does the effect survive post-training? Does it interact with RL later (Chapter 19)?

### What interviewers listen for

| Strong | Weak |
|---|---|
| Asks what decision the answer feeds into | Jumps to an experiment immediately |
| Names confounds and designs controls for them | Compares "with" vs "without" only |
| Matches compute and tokens | Adds data without removing any |
| Plans for noise: seeds, interval, paired evaluation | One run per arm |
| Uses small-scale proxies and checks the trend | Proposes a single huge run |
| States in advance what each result would mean | Leaves interpretation for later |
| Mentions contamination and evaluation validity | Trusts benchmark numbers |

### More practice questions (with the key points of a good answer)

- **"After our RL stage, long-context performance dropped. Investigate."** Reproduce on a fixed long-context suite; check the RL data's length distribution and maximum response length; check whether the regression is retrieval or reasoning; compare checkpoints through RL to find when it starts; test a mixed RL/long-context data fix or a KL constraint; verify with seeds.
- **"Design an evaluation for sycophancy."** Define the behaviours (answer flipping, opinion tailoring, feedback bias); build paired prompts with and without user pressure; measure flip asymmetry with McNemar tests; control for legitimate updating (arguments with real evidence); validate the judge (Chapters 23, 27).
- **"Is our new attention variant better than GQA?"** Matched tokens and parameters; tuned learning rates for each; several scales with scaling fits; long-context recall and quality evaluations; throughput and memory on real hardware (Chapter 11).
- **"How would you detect whether a model was trained on our benchmark?"** Perturbation gaps, guided completion, likelihood tests, time-split items; triangulate (Chapter 23).
- **"Our reward model accuracy went up but policy quality didn't improve. Why?"** Reward model accuracy on held-out pairs is not policy-level quality; over-optimisation; distribution shift between the pairs and policy outputs; length bias; evaluate with best-of-n gold curves (Chapter 18).

## 36.3 The paper discussion

Two formats: discussing a paper you chose (often your own), or a paper the interviewer sends in advance.

**For a paper sent in advance**, prepare:

1. **A two-minute summary**: problem, key idea, main evidence, result.
2. **What is convincing and why**: the strongest experiment.
3. **Critique**: baselines, confounds, noise, scale, evaluation validity, missing ablations (use the checklist in Chapter 34).
4. **Your next experiment**: the one that would most change your belief in the main claim.
5. **Connections**: related work and how this fits.

Interviewers value disagreement grounded in evidence ("the gain could be explained by the extra tokens; the paper doesn't control for that, and here is how I would"), not reflexive criticism. Acknowledge what the paper does well.

**For your own paper**, expect depth: why each design choice, what you tried that failed, what reviewers objected to, what you would do differently with ten times the compute, what you learned that is not in the paper. Know your numbers.

## 36.4 The research presentation

A 30–45 minute talk on your work, followed by questions. Structure for a mixed audience:

1. **The problem and why it matters** (3–5 minutes): one slide each for context, the gap and the question.
2. **The key idea** (5 minutes): intuition first, then the formal version.
3. **Evidence** (10–15 minutes): the two or three most important results, each with the setup needed to trust it, error bars and the comparison that matters.
4. **What did not work and limits** (3–5 minutes): this builds credibility.
5. **What's next** (2–3 minutes): including how it connects to the team you are interviewing with.
6. **Your contribution**: in collaborative work, say clearly which parts were yours.

Practise with a timer and with someone who will interrupt. Prepare backup slides for likely questions (ablations, compute, failure cases). For each result slide, be ready to answer "how noisy is this?" and "what would happen at larger scale?".

If your past work is not in LLMs, connect it explicitly: the methods (experiment design, evaluation, scaling, optimisation) transfer, and you should show you know the LLM-specific literature.

## 36.5 Take-home research tasks

Typical tasks: fine-tune or RL-train a small model for a capability and analyse the result; investigate a behaviour of an open model; reproduce a small result; design and run an evaluation. Usually a few hours to a few days, with limited compute.

**What graders look for:**

| Criterion | What good looks like |
|---|---|
| Problem framing | Clear question, stated hypotheses and success criteria |
| Baselines and controls | A sensible baseline, tuned fairly; at least one control |
| Rigour | Seeds or intervals, evaluation sized for the claims, no leakage |
| Analysis | Looks at examples, not only aggregates; error analysis; surprises investigated |
| Honesty | Limitations and negative results reported; claims match evidence |
| Communication | A short report a busy reader can follow: question, answer, evidence, limits, next steps |
| Code | Runnable, readable, with a README; not over-engineered |

**Advice**: spend the first 10% of the time planning and the last 20% writing. A smaller, careful project beats an ambitious, sloppy one. If something does not work, report it and explain what you learned; graders often weigh judgement over results. Respect the time limit, and say what you would do with more time.

## 36.6 Depth and coding for research roles

- **Coding** (Chapter 35): expect to implement from scratch, in PyTorch or NumPy, things such as multi-head attention with a KV cache and causal mask, RoPE, a transformer block, top-p sampling, beam search, BPE, LoRA, the DPO, PPO-clip and GRPO losses, an MoE router with load balancing, numerically stable softmax and cross-entropy with masking, and the unbiased pass@k estimator. Talk through shapes, test as you go, and handle edge cases.
- **Depth** (Chapter 37): expect follow-ups that go one level deeper each time: "why does that work?", "what breaks at scale?", "what's the evidence?". It is fine to say "I don't know, but here is how I'd reason about it" — and then reason.

## 36.7 Questions to ask your interviewers

Research roles vary enormously in what you actually do. Ask:

- What did the last person in this role work on in their first six months? What came of it?
- How are research directions chosen? How much is bottom-up?
- What compute does the team have, and how is it allocated between people?
- How does research reach production models? What is the path for a result?
- What is the publication policy?
- How is impact evaluated at performance reviews?

## Pitfalls

- Answering a design question with a single "with vs without" experiment.
- Forgetting compute matching, seeds and contamination.
- Presenting a paper with only praise or only criticism.
- Over-running the talk; no slide on what did not work.
- Take-homes with impressive scope and no baseline, no error bars and no limitations section.
- Not knowing the numbers in your own papers.

## Interview questions

**1. Design an experiment to test whether a data filter improves downstream reasoning.** Clarify the decision; hypotheses including more-tokens and contamination; matched-token anneals or small pretraining runs with and without the filter plus a control filter of equal strength; two scales, three seeds; reasoning, regression and per-domain loss evaluations; decision rules stated in advance.

**2. Walk me through your most important paper as if I were a sceptical reviewer.** Problem, idea, strongest evidence with error bars, what you tried that failed, limitations, what you would do next, your specific contribution.

**3. Critique this paper's main table.** Baselines re-run or copied? Tuned? Compute matched? Seeds and intervals? Consistent across tasks and scales? Contamination? What ablation is missing?

**4. You have one week and 64 GPUs to test an idea. How do you plan?** Day 1: proposal with a kill criterion and the smallest experiment; days 2–4: fast small-scale runs with seeds and baselines; day 5: confirm at a larger scale if justified; days 6–7: analysis and memo.

**5. What would make you abandon a research direction?** A pre-stated kill criterion met, effects that vanish with better baselines or more seeds, scaling trends that flatten, or a cheaper method achieving the same result.

## Practice

1. Answer three research-design questions from 36.2 aloud in 20 minutes each, recording yourself; compare with the "strong" column of the table.
2. Pick a recent paper; prepare the five-part discussion in 36.3 and present it to a friend in 15 minutes.
3. Build a 30-minute talk on your best project, with a "what didn't work" slide and three backup slides; rehearse it twice with interruptions.
4. **Research exercise.** Do a self-imposed take-home: in 8 hours, investigate one behaviour of an open model (for example sycophancy under pushback or a long-context failure), and write a two-page report with the template in Chapter 34. Ask someone to grade it with the rubric in 36.5.

## Read next

- Chapter 34 (research craft), Chapter 35 (coding), Chapter 37 (depth questions), Chapter 38 (behavioural).
- John Schulman, "An Opinionated Guide to ML Research" (2020).
- Simon Peyton Jones, "How to give a great research talk" and "How to write a great research paper" (talks, online).
- The research blogs of the teams you interview with: their recent papers are the best predictor of their questions.

## Done when

You can answer a research-design question with clarification, hypotheses including confounds, a controlled and compute-matched design, a noise plan, decision rules and extensions. You can discuss a paper with a balanced critique and a concrete next experiment.

You can give a 30-minute talk on your work that includes limitations and your contribution, and answer questions on its numbers. You can scope and write up a take-home research task that a grader would score highly on rigour and communication.
