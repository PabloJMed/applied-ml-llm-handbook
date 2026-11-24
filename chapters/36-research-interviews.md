# 36 · Research interviews

Part X · Research craft and interviews. Builds on Chapter 34 (research craft), Chapter 35 (coding rounds) and Chapter 37 (depth questions). Chapter 38 covers behavioural rounds, including research stories.

## Why it matters

Interviews for research scientist roles in LLMs test different things from applied ML loops. Besides coding and depth questions, you will usually face a **research design** round ("how would you test whether X improves Y?"), a **paper discussion** (yours or someone else's), a **research presentation** on your past work, and sometimes a **take-home research task**. These rounds test judgement: whether you ask the right question, design an experiment that answers it, anticipate confounds, read results sceptically and communicate clearly. This chapter explains what each round looks for and works through example answers.

The chapter starts with the shape of a typical research interview loop (36.1), then takes each round in turn: the research design round (36.2), the paper discussion (36.3), the research presentation (36.4) and take-home tasks (36.5). Section 36.6 covers the depth and coding rounds as they differ for research roles, and 36.7 the questions you should ask. It then works through a paper discussion of DPO (36.8), with the derivation interviewers expect on the whiteboard, and ends with the research-vision round that frontier labs use to test taste (36.9).

Research rounds probe judgement more than knowledge: what you would test first, what result would change your mind, which confounder worries you, and what you would do with a disappointing result. The best preparation is practising out loud on real questions, and this chapter gives the structure for doing so. The numbers in its worked answers (budgets, standard errors, the DPO checks) are computed in `code/ch36_check.py`.

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

*Minimum effect worth acting on.* Replacing 20% of the mid-training mixture is not free: it displaces other data. Decide before running anything what gain would justify it, for example +1 point on an aggregate of out-of-domain reasoning suites totalling about 3,000 items. Then check that the evaluation can see that. At 60% accuracy, one model's aggregate score has a standard error of √(0.6 × 0.4 / 3,000) = 0.89 points, so the difference between two independently scored models has a standard error of 1.26 points, too large to resolve a 1-point gain. Scoring both models on the same items and analysing the paired differences helps: if 15% of items change outcome between the two models, the standard error of the difference is about √(0.15 / 3,000) = 0.71 points. So the design must use paired evaluation, and either more items or the 8B run, to decide.

*Budget.* Take anneals of 2B tokens at 400M parameters and 6B tokens at 1.4B. With C = 6ND (Chapter 13), one anneal costs 4.8 × 10¹⁸ and 5.0 × 10¹⁹ FLOPs; three arms × three seeds × two sizes total 5.0 × 10²⁰ FLOPs, about 350 GPU-hours at 40% MFU on an H100 (Chapter 34.7 shows the conversion). The 8B confirmation dominates: two 100B-token anneals (baseline and maths) cost 9.6 × 10²¹ FLOPs, about 6,700 GPU-hours. The small-scale phase is therefore about 5% of the budget, which argues for being generous with seeds there: it is the cheap part.

*What the small scales can and cannot say.* At 400M and 1.4B parameters most reasoning benchmarks sit near chance, so "general reasoning" at those sizes has to be read from proxies: per-domain held-out loss on reasoning text, or the probability assigned to correct answers. Those proxies tell you about direction, not about the size of the effect at 8B. Transfer from maths to other reasoning may only appear at larger scale, or may appear at small scale and vanish. Say this explicitly in the interview: the small-scale phase decides whether the 8B run is worth doing and with what proportion; only the 8B run decides inclusion.

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
- **"Our coding-agent RL reward keeps rising, but users say the model got worse. What is going on?"** Suspect reward hacking first (Chapter 27): sample high-reward trajectories and read the diffs for edited tests, special-cased inputs or suppressed errors; compare training reward with the pass rate on held-out tasks graded in a clean sandbox with hidden tests; check whether the agent reads git history or other leaked answers. Then harden the verifier (Chapter 33.11), re-run from the last checkpoint before the divergence, and add the hack you found to a regression suite of known exploits.
- **"Design an RL environment and verifier for a new skill, say writing SQL against a warehouse."** Tasks with executable ground truth (compare result sets, not query strings); hidden test databases so the agent cannot fit to one dataset; checks for degenerate solutions (a query that returns the expected rows by hard-coding them); a difficulty curriculum so groups are not all-pass or all-fail (Chapter 19.4); throughput per sandbox; and a held-out set of environments for evaluation.
- **"Should we spend inference compute on best-of-N or on longer thinking?"** Compare at matched compute (tokens or FLOPs per problem) on curves, not points; best-of-N needs a verifier or reward model, and its gain is bounded by verifier quality (with a perfect verifier and a 30% per-sample success rate, best-of-8 solves 1 − 0.7⁸ = 94%, while majority voting gains far less on hard problems); longer thinking needs no verifier but saturates; the answer usually differs by difficulty, so report it per difficulty bucket (Chapters 13.9 and 19.8).
- **"How would you evaluate a research agent before we let it run overnight?"** End-state task success on held-out tasks, trajectory checks for forbidden actions, cost and wall-clock per task, pass^k over repeated runs rather than one demo, and behaviour under tool failures and injected instructions in retrieved content (Chapters 25 and 33).
- **"How would you test whether a model behaves differently when it believes it is being evaluated?"** Pairs of prompts that differ only in evaluation cues (a benchmark-style format, an explicit "this is a test") against realistic deployment-style versions; measure the behaviour difference with a paired test (McNemar on discordant pairs); read reasoning traces for verbalised awareness; and test for sandbagging by checking whether capability scores rise under elicitation, such as fine-tuning on a few examples (Chapter 27.7).

### A second worked design: long-context performance dropped after RL

*Prompt.* "After our RL stage, the model's long-context score fell from 72% to 64%. Investigate."

*Clarify.* Which suite, how many items, which harness, and was anything else changed between the two checkpoints (chat template, system prompt, sampling settings)? Suppose 400 items, the same harness, temperature 0.7.

*Is the drop real?* The standard error of the difference between two independent scores on 400 items is √(0.72 × 0.28 / 400 + 0.64 × 0.36 / 400) = 3.3 points, so an 8-point drop is about 2.4 standard errors: probably real, but at temperature 0.7 a single sample per item adds sampling noise. Re-run both checkpoints with 3–5 samples per item and compare per-item mean scores, paired.

*Hypotheses.*
- H1: the RL data were short, so long-context skills drifted from disuse.
- H2: the maximum response or context length during RL truncated long inputs, and the model learned to ignore distant context.
- H3: removing the KL penalty let the policy move far from the reference in directions RL never measured.
- H4: a format change, not a capability change: RL taught a reasoning style whose final answer the long-context grader fails to parse.
- H5: an evaluation change between the runs.

*Localise before intervening.* Score intermediate RL checkpoints to see when the drop begins (gradually from the start points to H1 or H3; a step change points to a data or configuration change at that step). Split the suite into retrieval items (find a fact) and reasoning items (combine facts across the document): H1 and H2 hit both, H4 shows up as answers that are right in the text but marked wrong. Read 30 failures before running anything else; H4 and H5 are usually visible within minutes.

*Interventions, at matched compute.* Mix 5–10% long-context prompts into the RL data; restore a small KL penalty or lower the learning rate; for H4, fix the answer extraction and re-score. Run each from the same pre-RL checkpoint with 2–3 seeds, and track both the RL reward and the long-context score.

*Decision.* If the long-context mix restores the score without lowering the RL reward beyond its seed noise, adopt it and add the long-context suite to the RL monitoring dashboard. If H4 explains it, fix the grader and report that no capability was lost. If only a strong KL restores it at a cost in reward, report the trade-off curve and let the product priorities choose the point.

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

**A slide plan for a 30-minute talk.** About one slide per two minutes, with the time weighted towards evidence:

| Minutes | Slides | Content |
|---|---|---|
| 0–3 | 1–2 | The problem, why it matters, and the one question the talk answers |
| 3–5 | 3 | The gap: what existing methods do and where they fail, with one number |
| 5–10 | 4–6 | The key idea: intuition with a picture, then the formal version |
| 10–22 | 7–11 | Evidence: the main result with error bars, the ablation that isolates the idea, the scaling or generalisation result |
| 22–25 | 12 | What did not work and the limits |
| 25–28 | 13 | Next steps, connected to the team you are interviewing with |
| 28–30 | 14 | Summary: the claim, the evidence, your contribution |
| | Backup | Hyperparameters, compute, extra ablations, failure cases |

**Handling hard and hostile questions.** Interrupting questions are often a test of how you handle pressure, not an attack. Three exchanges show the pattern: acknowledge, answer with evidence, and say what you would do if the evidence is missing.

- *"Isn't this just more compute?"* "That was our main worry too. Slide 9 compares the two at matched FLOPs: the gap is 1.4 points with an interval of ±0.5 across five seeds. At matched wall-clock it is smaller, about half, because our method is slower per step."
- *"Your baseline looks under-tuned."* "We gave it the same sweep: five learning rates and three warm-up lengths, the same as our method; the backup slide has the curves. If you have a setting in mind, I'd like to hear it, because that is the most likely way the result is wrong."
- *"I don't believe this generalises beyond your benchmark."* "That's fair; we have two tasks, and both are in the same domain. The experiment that would convince me is the held-out task family on slide 12's list, which we have not run."

Do not argue past the evidence, and do not concede a point you have data against; "I don't know, here is how I would find out" is a strong answer when it is true.

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

**Using LLM assistants.** Read the instructions on assistant use and follow them exactly; if they are silent, ask. Where assistants are allowed, say in the report what you used them for (writing plotting code, drafting the related-work paragraph) and what you checked yourself. Graders are judging your judgement, so the analysis, the design decisions and the conclusions must visibly be yours, and every number must come from code you ran and understand. An undisclosed, assistant-written report that the grader later probes in a follow-up interview is the fastest way to lose an offer.

**What a strong report looks like.** Two pages plus appendices, in this order:

1. **Question and answer** (three or four sentences): what you tested and what you found, with the key number and its interval. "Fine-tuning on 2,000 synthetic pushback dialogues cut answer flips under pushback from 31% to 12% (95% CI 8–16%) with no change on the held-out QA set (±1.5 points)."
2. **Setup** (half a page): model, data, the baseline and controls, compute, seeds, evaluation, each in a sentence.
3. **Results** (one figure and one table): the main comparison with intervals, and the control.
4. **Analysis** (half a page): examples of successes and failures, what explains them, what surprised you.
5. **Limitations and what would change my mind** (a short paragraph).
6. **Next steps with more time** (three bullets).
7. **Appendix**: hyperparameters, extra results, how to run the code.

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

## 36.8 A worked paper discussion: DPO

The interviewer asks you to discuss *Direct Preference Optimization* (Rafailov et al., 2023), which you were told to read beforehand. A strong 15-minute discussion has five parts.

1. **The claim in two sentences.** RLHF's KL-regularised reward maximisation has a closed-form optimal policy. So the reward can be written in terms of that policy and the reference model, and the preference likelihood can be optimised directly with a classification-style loss. This replaces reward-model training and PPO with one supervised objective that is simpler, more stable and cheaper.
2. **The key step, written out.** Do this on the whiteboard; it takes three minutes and is the part interviewers score most.
   - RLHF maximises reward while staying close to a reference policy π_ref, with strength β > 0, for each prompt x:

$$
\max_{\pi}\ \mathbb{E}_{y\sim\pi}\left[r(x,y)\right] - \beta\,\mathrm{KL}\left(\pi\,\|\,\pi_{\mathrm{ref}}\right)
$$

   - Define π*(y|x) = π_ref(y|x) exp(r(x, y)/β) / Z(x), where Z(x) = Σ_y π_ref(y|x) exp(r(x, y)/β) normalises it. Substituting r = β log(π*Z/π_ref) into the objective turns it into −β KL(π ‖ π*) + β log Z(x). The second term does not depend on π and the first is maximised (at zero) by π = π*, so π* is the optimal policy. Taking logs and rearranging expresses the reward through the optimal policy:

$$
r(x,y) = \beta\log\frac{\pi^{*}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x)
$$

   - The Bradley–Terry model says the chosen response y_w beats the rejected y_l with probability σ(r(x, y_w) − r(x, y_l)), where σ is the logistic function. Only the difference of rewards enters, so β log Z(x) cancels. Replacing π* by the trainable policy π_θ and maximising the likelihood of the observed preferences gives the DPO loss:

$$
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_{\theta}(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)
$$

   - **What the gradient does.** Write r̂(y) = β log(π_θ(y|x)/π_ref(y|x)) for the implicit reward. The gradient of the loss is −β σ(r̂(y_l) − r̂(y_w)) [∇ log π_θ(y_w|x) − ∇ log π_θ(y_l|x)]: it raises the chosen response's log-probability, lowers the rejected one's, and weights each pair by σ(r̂_l − r̂_w), the probability the current implicit reward model assigns to the *wrong* ordering. At the start of training π_θ = π_ref, every implicit reward is 0, the weight is σ(0) = 0.5 and the loss is ln 2 = 0.693 for every pair. A pair already ranked correctly by a margin of r̂_w − r̂_l = 0.15 (a log-ratio margin of 1.5 at β = 0.1) has loss 0.621 and weight 0.46; well-learned pairs fade out of the gradient.

3. **The evidence and its limits.**
   - Experiments: controlled sentiment generation, summarisation and single-turn dialogue.
   - Models: up to a few billion parameters.
   - Evaluation: largely GPT-4 win rates against references or PPO.
   - Questions to raise: the judge's biases (length, position); the scale; the fact that preference data was collected offline from other policies; and how well PPO baselines were tuned.
4. **What later work found.**
   - Both the chosen and the rejected log-probabilities often fall during training; only their difference rises (likelihood displacement). It is tempting to blame the offline data, but the mechanism is in the objective: the loss constrains only the margin, so it can be satisfied by pushing both responses down as long as the rejected one falls faster. Razin et al. (2024) relate the size of the effect to how similar the chosen and rejected responses are, and show that the probability mass can move to responses outside the data, sometimes unwanted ones (Chapter 18.6).
   - It can exploit length, because summed log-probabilities favour longer chosen responses.
   - It can overfit when preferences are nearly deterministic, since the optimum then pushes the margin to infinity regardless of β, which motivated IPO and other variants.
   - The data are offline, and online or iterative DPO, with fresh samples from the current policy, often closes much of the gap to online RL.
   - Reference-free variants (SimPO) change the implicit reward.
   - Mention one or two with what each changes, not a list of names.
5. **Your own follow-up.** Propose one experiment. For example: at matched preference data and compute, compare offline DPO, iterative DPO and PPO at two scales, with length-controlled win rates and a held-out human evaluation, to separate the effect of on-policy data from the effect of the objective.

**What interviewers listen for:**

- **Precise mathematics.** Can you derive the loss, not only name it?
- **Evidence-weighing.** Do you separate what the paper shows from what it claims?
- **Awareness of later work.** Do you know what came after, and why?
- **A testable follow-up.** Do you propose an experiment, not just an opinion?

Prepare three papers to this depth in your area. Include one you disagree with: explaining a disagreement precisely is one of the strongest signals in a research loop.

## 36.9 The research-vision round

Frontier labs often include a conversation that is neither design nor paper discussion: "What would you work on in your first six months here?", "Where do you think the field is going?", or "What is the most important open problem in your area?". It tests research taste (Chapter 34.2): whether you can pick a problem that matters, explain why now and why here, and turn it into a plan that would produce evidence quickly. It is the round where junior candidates most often give a list of fashionable topics, and where a structured answer stands out most.

### A structure that works

1. **A thesis** (two minutes): one claim about where the field is going and the evidence for it. A thesis is falsifiable; "agents are important" is not a thesis.
2. **The problem it implies** (two minutes): the bottleneck the thesis creates, stated precisely enough that someone could disagree.
3. **Why here** (one minute): what this lab has that makes the problem tractable (its models, data, infrastructure, an existing team effort). Read their recent papers and blog posts first.
4. **The first six months** (three minutes): month 1 reproduce and instrument; months 2–3 the cheapest experiments that could kill the idea, with criteria stated; months 4–6 scale the survivor or pivot.
5. **What would change your mind** (one minute), and how it connects to the team's roadmap.

### A worked answer: verifiers for agentic RL

*Thesis.* Over the past two years, the capability gains in reasoning and agentic coding have come mostly from RL against verifiable rewards, and the ceiling on that progress is increasingly set by the verifiers and environments rather than by the RL algorithm. The evidence I would cite: published comparisons of GRPO variants often differ by less than the choice of prompts and reward design moves results (Chapter 19.4), and reward hacking in coding environments (edited tests, leaked answers) is now a routine finding (Chapter 33.11).

*Problem.* We have no good way to measure how hackable a verifier is before we train against it. Teams find hacks after the fact, by reading trajectories once the reward has diverged from held-out performance.

*Why here.* This lab trains agentic models at scale and has the environment infrastructure, so it has both the problem and the data to study it: logs of past runs in which hacks were found.

*First six months.*
- **Month 1.** Collect the known hacks from past runs into a benchmark of exploit trajectories, and build a harness that replays a policy against an environment with a stricter, hidden grader.
- **Months 2–3.** Test one cheap idea with a kill criterion. An automated red-team policy, trained with RL to maximise reward *minus* the strict grader's score, should rediscover at least half of the known hacks within a small compute budget (say 200 GPU-hours per environment). If it finds fewer than a quarter, the approach is too weak, and I would fall back to static analysis of the verifier.
- **Months 4–6.** If it works, run the red-team before every new environment enters training, and measure whether hardened environments give a smaller gap between training reward and held-out pass rate in real runs.

*What would change my mind.* If held-out performance tracks training reward closely in most runs, hackability is not the bottleneck, and environment diversity is the better investment.

The answer works because it is specific, has numbers that create a decision, uses the lab's assets, and states its own falsification. An interviewer will push on one part, usually the thesis ("what if algorithms matter more than you think?"); answer with the evidence you would look at, not by restating the claim.

## Pitfalls

- Answering a design question with a single "with vs without" experiment.
- Forgetting compute matching, seeds and contamination.
- Presenting a paper with only praise or only criticism.
- Over-running the talk; no slide on what did not work.
- Take-homes with impressive scope and no baseline, no error bars and no limitations section.
- Not knowing the numbers in your own papers.
- Naming DPO without being able to derive it, or blaming every DPO pathology on offline data.
- A research-vision answer that is a list of topics rather than a thesis with a plan.
- Designs without a budget or a minimum effect size, so no result could change the decision.

## Interview questions

**1. Design an experiment to test whether a data filter improves downstream reasoning.** Clarify the decision (include the filter in the next run or not) and the minimum gain that would justify the tokens it removes. State the hypotheses, including the boring ones: more tokens of anything, contamination, a style change. Then run matched-token anneals or small pretraining runs with and without the filter, plus a control filter that removes the same share of data at random or by an unrelated score. Use two scales and three seeds; evaluate reasoning, a regression suite and per-domain loss; state the decision rules in advance. Put a number on the noise: on a 3,000-item reasoning aggregate at 60%, two independent scores differ with a standard error of 1.3 points, so a 1-point effect needs paired evaluation. What I would not do: compare one filtered run with last quarter's baseline. Follow-up: "the filter helps at 400M but not 1.4B?" Run the third size before deciding; a shrinking effect is the most common outcome.

**2. Walk me through your most important paper as if I were a sceptical reviewer.** Problem, idea, strongest evidence with error bars, what you tried that failed, limitations, what you would do next, and your specific contribution. Lead with the weakest point before they find it ("our main result is at one scale; here is why I still believe it"), and know the numbers cold: the effect, its interval, the number of seeds, and the compute. When not to defend: if they find a real flaw, agree, say how you would fix it, and estimate how much it would change the result. Follow-up: "what would you do with ten times the compute?" Name the one experiment it buys and the result that would change your conclusion.

**3. Critique this paper's main table.** Ask whether baselines were re-run or copied, and tuned with the same budget; whether compute and data were matched; whether there are seeds and intervals; whether the gains are consistent across tasks and scales; whether contamination was checked; and which ablation is missing. Then quantify: a 2-point gain on a 500-item benchmark at 70% is within one standard error of a difference of two independent scores (2.9 points), so ask whether the evaluation was paired or repeated. End with the single experiment that would most change your belief. Follow-up: "anything good about it?" Name the strongest result and why it is convincing; a critique that finds nothing good reads as reflexive.

**4. You have one week and 64 GPUs to test an idea. How do you plan?** Start with the budget: 64 × 168 = 10,752 GPU-hours. With the numbers of Chapter 34.7, a four-size ladder from 125M to 1.4B with three learning rates and two seeds costs about 1,350 GPU-hours (about 21 hours on 64 GPUs if the runs pack well), and one 7B, 140B-token confirmation costs about 4,130 GPU-hours (about 65 hours). Both fit, using half the week's compute, which leaves room for reruns. Plan: day 1, a proposal with a kill criterion and the smallest experiment; days 2–3, the ladder with baselines and seeds; days 3–6, the 7B run only if the ladder's trend justifies it; day 7, analysis and memo. What I would not do: start the 7B run on day 1, or skip the baseline seeds. Follow-up: "the ladder is ambiguous on day 3?" Spend the remaining compute on more seeds at the two cheapest sizes rather than the 7B run.

**5. What would make you abandon a research direction?** A pre-stated kill criterion met; effects that vanish with better baselines or more seeds; scaling trends that flatten; or a cheaper method achieving the same result. Make the criterion numeric in advance: "stop if the 125M difference is below one seed SD and the 350M difference is smaller still". Distinguish a dead idea from a bad test: before killing, check that the experiment could have detected the effect (its power), since an underpowered null is not evidence of absence. Follow-up: "have you ever kept going past a kill criterion?" A good answer names a case and the new evidence that justified it.

**6. What would you work on in your first six months here?** Give a thesis, the problem it implies, why this lab is the place to solve it, a month-by-month plan with a cheap kill test in months 2–3, and what would change your mind (36.9). Tie it to the lab's assets and recent work, and give one number that drives a decision, such as the compute for the first kill test or the effect size that would justify scaling. What not to do: list five fashionable topics. Follow-up: "why hasn't someone already done this?" Say what is new (data, scale, a tool) or admit it may be in progress and say how your angle differs.

**7. How would you design a verifier for RL on a coding task so the policy cannot hack it?** Grade in a fresh sandbox: apply only the agent's diff to the non-test files of a clean checkout, restore the original tests, and run hidden tests the agent never saw; remove git history and network access so answers cannot leak; flag diffs that touch tests or hard-code outputs; and monitor the gap between training reward and the pass rate on a held-out set under a stricter grader. Treat the verifier as an adversarial target: before training, run a red-team policy against it. Follow-up: "how do you know it worked?" The training-reward versus held-out-pass-rate gap stays flat, and sampled high-reward trajectories read as genuine solutions (Chapter 33.11).

**8. Best-of-N sampling or longer thinking: how would you decide where to spend inference compute?** Compare them at matched compute on full curves, not single points, and per difficulty bucket. Best-of-N needs a verifier and is bounded by its quality: with a perfect verifier and a 30% per-sample success rate, best-of-8 solves 94% of problems, while majority voting helps much less when the correct answer is rarely the most common one. Longer thinking needs no verifier but saturates, and costs latency because it is sequential. Typically, longer thinking helps more on medium problems and sampling with a good verifier on hard ones. Follow-up: "what if the verifier is a learned reward model?" Then best-of-N over-optimises it as N grows, so plot the true score against N and stop before it turns down (Chapters 13.9 and 18.5).

**9. How would you test whether a model behaves differently when it thinks it is being evaluated?** Build pairs of prompts that differ only in evaluation cues and compare behaviour pairwise. With 500 pairs, if 40 behave safely only in the evaluation-styled version and 15 only in the deployment-styled one, McNemar's statistic is (40 − 15)² / 55 = 11.4, p < 0.001. Add reading of reasoning traces for verbalised awareness, and test for sandbagging by checking whether capability rises under stronger elicitation such as light fine-tuning (Chapter 27.7). Follow-up: "what if the model hides it?" Use realistic deployment-like settings and interpretability probes, and treat behavioural tests as a lower bound.

## Practice

1. Answer three research-design questions from 36.2 aloud in 20 minutes each, recording yourself; compare with the "strong" column of the table.
2. Pick a recent paper; prepare the five-part discussion in 36.3 and present it to a friend in 15 minutes.
3. Build a 30-minute talk on your best project, with a "what didn't work" slide and three backup slides; rehearse it twice with interruptions.
4. **Research exercise.** Do a self-imposed take-home: in 8 hours, investigate one behaviour of an open model (for example sycophancy under pushback or a long-context failure), and write a two-page report with the template in Chapter 34 or the skeleton in 36.5. Ask someone to grade it with the rubric in 36.5.
5. Derive the DPO loss from the KL-regularised objective on paper, then check that the loss is ln 2 when π_θ = π_ref.
6. Write a five-minute research-vision answer for a lab you would like to join, in the 36.9 structure, with one numeric kill criterion.

<details><summary>Hint and expected result</summary>

Item 1: a strong answer covers all six steps of 36.2 in 20 minutes and states at least one control and one decision rule before any result; if your recording spends more than five minutes before the first hypothesis, you are over-clarifying. Item 2: the most common weakness is a critique without a proposed experiment. Item 3: the talk should fit the 14-slide plan in 36.4; if the evidence section is under 10 minutes, cut the background. Item 4: expect a modest or null result; the report is graded on design and honesty, so a clear negative result with a control is a good outcome. A typical 8-hour budget: 1 hour planning, 4 hours running, 1 hour analysis, 2 hours writing. Item 5: at π_θ = π_ref the bracket is 0, σ(0) = 0.5 and −log 0.5 = 0.693. Item 6: read it to someone outside your area; if they cannot repeat the thesis in one sentence, it is not yet a thesis.

</details>

## Read next

- Chapter 34 (research craft), Chapter 35 (coding), Chapter 37 (depth questions), Chapter 38 (behavioural).
- John Schulman, "An Opinionated Guide to ML Research" (2020).
- Simon Peyton Jones, "How to give a great research talk" and "How to write a great research paper" (talks, online).
- The research blogs of the teams you interview with: their recent papers are the best predictor of their questions.

## Done when

- Given a research-design question, you produce, in 20 minutes and aloud, the clarification, hypotheses with confounds, a compute-matched design with a control, a noise plan with one computed standard error, decision rules and extensions.
- Without notes, you derive the DPO loss from the KL-regularised objective in 5 minutes, write its gradient weight σ(r̂_l − r̂_w), and explain why log Z cancels.
- You explain likelihood displacement as a property of the margin objective, not only of offline data.
- You compute the budget of a one-week, 64-GPU project (10,752 GPU-hours) and say what fits from the 34.7 table.
- You can give a 30-minute talk on your work in the 36.4 slide plan, including limitations and your contribution, and handle three hostile questions without conceding points you have evidence against.
- You give a five-minute research-vision answer with a thesis, a six-month plan and a numeric kill criterion.
- You have written one two-page take-home report in the 36.5 format and had it graded against the rubric.

---

[← 35 · The ML coding interview](35-the-ml-coding-interview.md) · [Contents](../README.md#contents) · [37 · ML depth question bank →](37-ml-depth-question-bank.md)
