# 34 · Research craft

Part X · Research craft and interviews. Draws on Chapter 2 (statistics), Chapter 13 (scaling laws and small-scale proxies) and Chapter 23 (evaluation statistics). Chapter 36 covers how research craft is tested in interviews.

## Why it matters

Two researchers with the same knowledge and compute produce very different output. The difference is craft: picking problems worth solving, running experiments that answer the question asked, telling signal from noise, iterating fast, and writing results others can trust and build on. None of this appears in papers, and most of it is learned by making expensive mistakes. This chapter collects the habits that separate productive research scientists from busy ones, with simulations that show why the statistical habits matter.

Numbers in this chapter come from `code/ch34_check.py` (seconds).

The chapter follows a research project from start to finish. It begins with what craft consists of (34.1) and how to choose a problem worth working on (34.2), then covers reading papers efficiently (34.3) and reproducing results before building on them (34.4). Sections 34.5 to 34.8 are the experimental core: designing experiments that can answer the question, handling noise between seeds, budgeting compute for fast iteration, and the infrastructure that keeps results trustworthy. The chapter then covers writing (34.9), collaboration and integrity (34.10), and a worked project from question to conclusion (34.11), and ends with the habits specific to LLM research: proxies that move at small scale, uncertain scaling extrapolations, training-run triage and multiple comparisons (34.12).

The skill underneath all of it is turning a vague idea into an experiment whose outcome would change your mind. Most wasted research time goes into experiments that could not have failed, or whose results could not be distinguished from noise, and every section is in some way a defence against that.

## Key concepts

- **Research taste**: judgement about which problems are important, tractable and suited to you. [Hamming, "You and Your Research"](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html), [Schulman, "An Opinionated Guide to ML Research"](http://joschu.net/blog/opinionated-guide-ml-research.html).
- **Reading papers**: structured reading and critical checking. [Keshav, "How to Read a Paper"](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf).
- **Ablations and baselines**: isolating what causes an improvement; tuning baselines as hard as methods. [Melis et al., "On the State of the Art of Evaluation in Neural Language Models"](https://arxiv.org/abs/1707.05589).
- **Seeds and variance**: [Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks"](https://arxiv.org/abs/2103.03098), [Agarwal et al., "Deep RL at the Edge of the Statistical Precipice"](https://arxiv.org/abs/2108.13264).
- **Experiment tracking**: [Weights & Biases](https://wandb.ai/), [MLflow](https://mlflow.org/).
- **Training recipes**: [Karpathy, "A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/), [Google's Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook).

## 34.1 What craft consists of

Three things, in roughly this order of importance:

1. **Taste**: choosing what to work on. A well-executed answer to an unimportant question is still unimportant.
2. **Rigour**: designing experiments whose results mean what you claim.
3. **Speed**: shortening the loop from idea to evidence, so you get many more chances to be right.

They trade off. Too much rigour too early slows exploration; too little later produces results that collapse under scrutiny. A useful pattern: explore fast and loosely, then confirm slowly and carefully before you believe or share anything.

## 34.2 Choosing problems

Ask four questions before starting:

- **Impact**: if this works, what changes? Who would use it? Would it change a decision at your lab?
- **Tractability**: is there a plausible path to a result in weeks, not years? What is the smallest experiment that would tell you if you are on track?
- **Edge**: why you? Unique data, compute, infrastructure, an insight others lack, or a skill combination.
- **Timeliness**: is the field about to need this, or has it moved on?

Habits that improve taste are mostly about making the decision explicit and cheap to revisit. Keep a running list of open problems and failed ideas, and revisit it monthly: ideas that failed for a reason that has since changed (more compute, a new model, a better evaluation) are often the best ones to retry. Before any experiment, write a one-page proposal stating the question, the hypothesis, the experiment that would falsify it, and what you would do with each outcome; if you cannot say what you would do differently after a negative result, the experiment is not yet worth running. Time-box exploration (two weeks is typical) and end the box with an explicit decision to continue, pivot or kill, rather than letting a project drift. **Kill ideas cheaply**: look for the fastest experiment that could prove the idea wrong, and run it first. And prefer problems that matter even if they fail, because a careful negative result on an important question is still useful, while a positive result on an unimportant one is not.

### A worked choice between two problems

Suppose you have two candidate projects for the next quarter:

- **A.** A new optimiser variant. A recent preprint reports 3% lower loss than AdamW at 125M parameters.
- **B.** Using your team's in-house quality classifier to reweight the mid-training data mixture.

Run each through the four questions. **Impact:** A could matter across every training run if real, but optimiser gains at small scale have a poor track record of surviving tuned baselines and larger scales (34.5), so the expected impact is the claimed gain times a low probability that it holds. B's ceiling is lower, but data-mixture improvements have tended to transfer across scales, and the result feeds a decision the team must make anyway. **Tractability:** both can be tested in weeks; B with the matched-token anneal design of Chapter 36.2. **Edge:** anyone can test A, and several groups probably are; only your team has the classifier and the data for B. **Timeliness:** A is crowded, and the answer may be public before you finish.

The decision is B as the main project, with A as a two-day side experiment with a written kill criterion: re-run the preprint's 125M setting with the AdamW learning rate tuned as carefully as the new method's, three seeds each, and stop if the gap falls below one seed standard deviation. That split is the pattern interviewers like to hear: the bulk of the time on the problem where you have an edge, and a cheap, bounded test of the exciting idea.

## 34.3 Reading papers

Three passes (Keshav): skim (title, abstract, figures, conclusions: five minutes, decide whether to continue); read for the argument (method, experiments, without proofs); and, for papers you will build on, a deep pass where you could re-derive and reproduce the key result.

**Checklist for the claims**:

| Question | Red flag |
|---|---|
| What exactly is compared? | Baseline from an older paper, not re-run; different data or compute |
| Were baselines tuned as hard as the method? | Method swept over many settings, baseline at defaults |
| How big is the effect relative to noise? | No seeds, no intervals, gains under a point |
| Does it hold across scales and tasks? | One model size, one benchmark |
| Is the evaluation sound? | Contamination, judge without controls, cherry-picked examples |
| What do the ablations show? | Ablations that remove everything at once |
| What is not reported? | Compute, failed settings, wall-clock cost |

Keep a short note per important paper: claim, evidence, your doubts, and what experiment would test it.

**Triage at 2026 volume.** Hundreds of machine-learning papers appear on arXiv every day, so the first pass has to be ruthless, and most researchers now use an LLM to summarise or critique papers during triage. It helps with speed and with unfamiliar notation. Its failure modes are specific: summaries repeat the abstract's claims with the abstract's confidence, drop the caveats that live in the appendix (the baseline that was not re-tuned, the one model size), and occasionally invent a number or a comparison that is not in the paper. Use the model to decide what to read, and to ask pointed questions ("which table supports the claim about scale? which baselines were re-run?"), always with a request to cite the table or section. Then check the main table and figures yourself before the claim enters your notes. A claim you have only from a summary is a rumour.

## 34.4 Reproducing results

Reproducing a result is the best way to understand it and the fastest way to find out whether it is real.

1. Start from the official code if it exists; run it unchanged and match the reported numbers before changing anything.
2. Otherwise, start from the smallest setting the paper reports.
3. Match everything: data and preprocessing, tokenizer, model size, optimiser and schedule, evaluation code, number of samples.
4. When numbers differ, bisect: swap components between your version and the reference one at a time.
5. Report what you matched and what you could not.

Many "failed replications" come from an unreported detail (a learning-rate warm-up, a data filter, an evaluation prompt). Many successful ones show smaller effects than reported: expect that.

## 34.5 Designing experiments

**Baselines.** Tune the baseline with the same effort and budget as the new method. Melis et al. (2017) showed that a carefully tuned LSTM beat several newer architectures that had been reported as improvements. The same pattern recurs with optimisers, architectures and fine-tuning methods.

**One change at a time**, at matched compute or matched tokens, with matched schedules (Chapter 7) and the same evaluation.

**Ablations** answer "which part of this matters?". One-at-a-time ablations from a baseline miss interactions. In the check script, component B does nothing on its own (−0.25 from the baseline, within noise) but adds a full point when combined with A:

| Design | What it says about B |
|---|---|
| Add B to the baseline | "No effect" (−0.25) |
| 2 × 2 factorial | No effect without A (+0.03), +1.00 with A |
| Remove B from the full system | Costs 1.00 |

For systems with several components, ablate by removing each from the full system (leave-one-out), or run a factorial design if components are few.

**Why the table shows two different numbers for "B alone".** The −0.25 comes from one run of the baseline and one run with B added; the +0.03 is the difference of two cell means of the factorial, each averaged over three runs. Both estimate the same true effect (zero), and both are within noise: with a run-to-run SD of 0.2, the difference of two single runs has an SD of 0.2 × √2 = 0.28, and the difference of two 3-run means 0.2 × √(2/3) = 0.16. The four cell means of the factorial make the interaction visible:

| | Without B | With B |
|---|---|---|
| Without A | 49.96 | 49.99 |
| With A | 51.06 | 52.06 |

Reading across rows: B adds +0.03 without A and +1.00 with A. Reading down columns: A adds +1.10 without B and +2.07 with B. The true values in the simulation are 0, 1, 1 and 2 points above the baseline of 50; the interaction (the extra point when both are present) is what a one-at-a-time ablation from the baseline cannot see.

**Small-scale proxies.** Most research questions can be asked at small scale first. Check that the effect has the same sign and similar size at two small scales, and fit a trend (Chapter 13) before spending large compute. Some effects only appear at scale; some disappear. A single small-scale result is a hypothesis.

**Controls.** Include a condition that should not work (shuffled labels, random data, a trivially simple method). If it does surprisingly well, your evaluation has a problem.

## 34.6 Noise, seeds and statistics

Training runs differ by seed: initialisation, data order, dropout, non-deterministic GPU kernels. For small models fine-tuned on small datasets, seed variance can exceed the effects people report.

**How many seeds?** Suppose a change truly improves a score by 0.3 points and the seed-to-seed standard deviation is 0.4:

| Seeds per arm | What you see |
|---|---|
| 1 | The new run looks better 70% of the time; with no real gain it would still look better 50% of the time |
| 3 | 16% power to detect the gain at p < 0.05 |
| 5 | 27% |
| 10 | 49% |

**Where the 70% comes from.** With one seed per arm, the observed difference of the two runs is normal with mean δ (the true gain) and standard deviation σ√2, because each run carries its own seed noise σ and the variances add. The new run looks better when the difference is positive:

$$
P(\mathrm{new} > \mathrm{old}) = \Phi\left(\frac{\delta}{\sigma\sqrt{2}}\right)
$$

where Φ is the standard normal CDF. Here δ/(σ√2) = 0.3/(0.4 × 1.414) = 0.53 and Φ(0.53) = 0.70. With no real gain the argument is 0 and the probability is 0.50: a single comparison is barely better than a coin.

**Where the seed count comes from.** To detect a gain δ with a one-sided test at level α and power 1 − β, comparing the means of n seeds per arm, the difference of means has standard error σ√(2/n), and it must clear the critical value z_α·SE while sitting z_β·SE below the true δ. Solving δ = (z_α + z_β)·σ√(2/n) for n gives

$$
n = \frac{2\,(z_{\alpha} + z_{\beta})^2\,\sigma^2}{\delta^2}
$$

With α = 0.05 one-sided (z = 1.645), 80% power (z = 0.84) and δ/σ = 0.3/0.4 = 0.75, n = 2 × 2.485² / 0.75² = 22.0 seeds per arm. The exact t-test needs one more, 23, which is the figure's number: the t distribution's heavier tails cost a seed at these sample sizes. The formula makes the scaling obvious: halve the effect and you need four times the seeds (88 at δ/σ = 0.375); double it and 5–6 suffice (5.5 at δ/σ = 1.5).

When the effect is comparable to the noise, a handful of seeds cannot establish it. Options: reduce the noise (larger evaluation sets, continuous metrics such as loss, paired evaluation on the same items, Chapter 23), increase the effect (test at a scale where it is larger), or run many seeds of cheap configurations. At minimum, run 2–3 seeds of the baseline once to know the noise floor of your setup.

![Power of a one-sided t-test at 5% to detect a true gain, by seeds per arm, for gains of 1.5, 0.75 and 0.375 times the seed-to-seed standard deviation](images/34-fig-1.png)

*Power of a one-sided t-test at 5% to detect a true gain, by seeds per arm, for gains of 1.5, 0.75 and 0.375 times the seed-to-seed standard deviation. The middle curve is the table's case (a 0.3-point gain with 0.4-point noise): about 23 seeds per arm reach 80% power.*

The curve gives a planning rule. Measure the seed noise first, then express the effect you care about in units of it. Effects above about 1.5 standard deviations are visible with 5–7 seeds per arm. Effects below about 0.75 need either many cheap runs or a less noisy measurement.

**Winner's curse.** In a sweep of 12 configurations that are all equally good, with evaluation noise of 0.5 points, the best observed score averages 70.81 against a true 70.00. Re-running the winner regresses to 70.00. Always confirm a sweep's winner with fresh seeds before reporting it, and report the sweep, not only its maximum.

The 0.81 is the expected maximum of 12 independent standard normal draws, 1.63, times the noise SD of 0.5. The expected maximum grows slowly with the number of configurations k:

| Configurations k | 2 | 5 | 10 | 12 | 30 | 100 |
|---|---|---|---|---|---|---|
| E[max] in units of σ | 0.56 | 1.16 | 1.54 | 1.63 | 2.04 | 2.51 |

For large k it approaches σ√(2 ln k), which overstates it at small k (2.23 for k = 12, against 1.63) but gives the right shape: the optimism grows like the square root of the log of the number of things you tried. For a live estimate, remember about 1.5σ for 10 configurations, 2σ for 30 and 2.5σ for 100. If configurations differ in truth, the bias is smaller but still positive, so the rule is unchanged: re-run the winner with fresh seeds, and judge it against the sweep's spread, not against the baseline's single run.

**Report**: means and intervals over seeds, the number of seeds, and all configurations tried.

## 34.7 Compute budgeting and iteration speed

Plan compute like money. For a project testing an idea on a four-size model ladder (125M to 1.4B parameters, about 20 tokens per parameter), three learning rates and two seeds each, then one 7B confirmation run on 140B tokens, at 40% MFU on one 989-TFLOP/s GPU:

| Phase | FLOPs | GPU-hours |
|---|---|---|
| Ladder sweep (24 runs) | 1.9 × 10²¹ | ~1,350 |
| 7B confirmation | 5.9 × 10²¹ | ~4,130 |

Spending about a quarter of the budget on the small-scale sweep is typical and usually worth it: it tells you whether the large run is worth doing and which settings to use.

**Where the table comes from.** Training compute is about 6ND FLOPs for N parameters and D tokens (Chapter 13). The ladder uses N = 125M, 350M, 760M and 1.4B with D = 2.5B, 7B, 15B and 28B tokens, so one pass over the ladder costs 6 × (125M × 2.5B + 350M × 7B + 760M × 15B + 1.4B × 28B) = 3.2 × 10²⁰ FLOPs, and three learning rates times two seeds make 1.92 × 10²¹. One GPU at 40% MFU of 989 TFLOP/s (the H100's dense bf16 peak) delivers 989 × 10¹² × 0.4 × 3,600 = 1.42 × 10¹⁸ FLOPs per hour, so the sweep is 1.92 × 10²¹ / 1.42 × 10¹⁸ ≈ 1,350 GPU-hours. The 7B run is 6 × 7 × 10⁹ × 140 × 10⁹ = 5.9 × 10²¹ FLOPs, about 4,130 GPU-hours. The sweep is 1,350 / 5,480 = 25% of the total. On newer hardware, replace 989 TFLOP/s with the dense bf16 (or fp8) peak of your GPU and the MFU you actually measure; the structure of the calculation is the interview answer.

**Iteration speed** is often the real constraint:

- Keep a **fast loop** (minutes) for debugging and sanity checks, a **medium loop** (hours) for most experiments, and a **slow loop** (days) only for confirmation.
- Overfit one batch before training anything.
- Automate the boring parts: launching sweeps, collecting results, plotting.
- Run several experiments in parallel; do not wait idle for one.
- Delete experiments that cannot change your decision.

## 34.8 Experiment infrastructure

- **Configuration files** for every run, committed with the code; no hand-edited constants.
- **Tracking**: log configuration, code version (git commit), data version, metrics, system statistics and artefacts for every run (Weights & Biases, MLflow, or your lab's system).
- **Naming**: runs named so you can find them in six months.
- **Determinism where cheap**: fixed seeds for initialisation and data order (same seed twice gives identical results in the check script; a different seed does not); accept that GPU kernels and distributed training are not bit-exact, and measure the resulting variance instead.
- **Evaluation as code**, versioned, run automatically on checkpoints (Chapter 23).

## 34.9 Writing

Writing is part of the research, not a report on it. Writing early exposes gaps in the argument and missing experiments.

**A research memo** (one to three pages) for every finished experiment cycle:

1. **Question** and why it matters.
2. **Answer** in one or two sentences, with the key number and its uncertainty.
3. **Evidence**: the main figure or table, with the setup needed to interpret it.
4. **What else could explain it**, and what you did to rule it out.
5. **Limitations** and what would change your mind.
6. **Next steps** and the decision you recommend.

Figures: label axes with units, show intervals, use the same scale for compared panels, and make the main point visible without reading the caption. Claims must match evidence: "improves" needs an interval that excludes zero; "generalises" needs several settings.

**Negative results** deserve a memo too. They save colleagues from repeating the work, and they are part of your track record of judgement.

## 34.10 Collaboration and integrity

- Agree on roles and authorship early and revisit when contributions change.
- Credit others' ideas and code explicitly.
- Report all configurations and seeds, not the best ones; report failed replications.
- Never adjust evaluations after seeing results to make a method look better; if you must change the evaluation, re-run every method.
- Share code and data where you can; reproducibility is a service to the field and to your future self.

## 34.11 A worked project: does a data curriculum help?

**Question.** Does ordering pretraining data from easy to hard (by a quality-classifier score, say) lower the final loss at a fixed token budget? The answer feeds a decision about the data loader for next quarter's runs.

1. **Read first.** Prior results on curricula for language-model pretraining are mixed. Benefits often appear early in training and shrink by the end, and some vanish once the learning-rate schedule is tuned. So the prior is a small effect, and the design must be able to see one.
2. **Choose the metrics.**
   - Primary: final loss on a fixed held-out set of 50M tokens from the target distribution. Loss is continuous and low-noise, so it suits small effects.
   - Secondary: five benchmarks, reported with intervals and not used for the decision unless they agree with the loss.
3. **Reproduce the baseline.** The random-order baseline at 125M parameters must match the team's known loss curve within seed noise before anything else is run.
4. **Measure the noise.** Run three baseline seeds at 125M. Suppose the final loss varies with a standard deviation of 0.003 nats. An effect of 0.005 nats is then 1.7 standard deviations: by the curve above, 5–6 seeds per arm give good power, 3 give about 50%.

   **But a standard deviation from three runs is itself very uncertain.** With 2 degrees of freedom, the 95% interval for the true σ runs from about 0.52 to 6.3 times the observed value (from the χ² distribution). If the true σ were at the top of that range, 0.019 nats, the same 0.005 effect would be a quarter of a standard deviation and 3 seeds per arm would give under 10% power. The senior move is to treat the first estimate as provisional: budget 5–10 baseline seeds for the noise floor if the runs are cheap (with 10 seeds the interval narrows to 0.69–1.83σ), or re-plan the seed count after the first arm has finished. Also check that the seeds vary everything they should. Noise sources can also be shared on purpose: running baseline and curriculum with the same initialisation seed pairs the runs, and a paired analysis then removes the part of the noise the two arms have in common. That is a deliberate variance-reduction trick, but only if the analysis compares the pairs; treating paired runs as independent misstates the interval.
5. **Design with a control for the obvious confound.**
   - A curriculum changes not only the order but also which data the model sees during the final learning-rate decay. Add a third arm that keeps random order but places the hardest data in the last 10% of tokens.
   - If that arm matches the full curriculum, the effect comes from the end of training (an annealing effect, Chapter 16), not from ordering.
6. **Budget.**
   - Use three sizes (125M, 350M, 760M) at 20 tokens per parameter, with 3 seeds each. Two arms cost 5.2 × 10²⁰ FLOPs, about 360 GPU-hours at 40% MFU; the third arm brings it to about 540.
   - The largest single run takes about 49 GPU-hours, so the whole study fits in a week on a small cluster.
7. **Set the kill criteria in advance.** Stop if the 125M difference is below one seed standard deviation and the 350M difference is smaller still. A shrinking effect with scale is the most common outcome and is a result worth reporting.
8. **Write it up.**
   - Loss against compute for each arm and size.
   - The effect with intervals at each size, and the fitted trend.
   - The control arm's result.
   - A one-paragraph recommendation that names its scope ("up to 760M parameters and 15B tokens").

The plan takes a page and about an hour to write. It will save weeks of runs that could not have answered the question.

## 34.12 Craft specific to LLM research

Most of this chapter applies to any empirical ML. Four habits are specific to LLM research, and lab interviewers probe each.

### Measure what moves at small scale

Small models score at chance on most capability benchmarks, so a benchmark comparison at 125M parameters measures noise. A four-option multiple-choice benchmark has a chance score of 25%, and with 1,000 questions the standard error at that level is √(0.25 × 0.75 / 1,000) = 1.4 points; two small models at 26% and 27% are indistinguishable. Use quantities that move smoothly with scale: held-out loss or bits per byte on text from the target domain (Chapter 23.4), or continuous versions of the benchmark, such as the probability the model assigns to the correct answer in a cloze format, which rises long before accuracy does (Chapters 13.8 and 23). Check that the proxy agrees in sign with the real benchmark at the largest size where the benchmark is above chance, and say so when you report it.

### Treat scaling-law extrapolations as uncertain

A ladder of four model sizes fits a curve such as L(C) = a·C^(−b) + e, and the fit is then used to predict a run 10–100 times larger. Small errors in the exponent grow with the extrapolation: an error of 0.01 in b changes the predicted reducible loss at 100× the largest ladder compute by a factor of 100^(−0.01), about 4.5%, which can be larger than the effect being studied. Two sources of error dominate in practice. One is statistical, and a bootstrap over seeds or ladder points gives the interval of the prediction. The other is systematic: a learning rate or batch size tuned at one size and not re-tuned at the others bends the ladder. Report the prediction with its interval, re-tune the main hyperparameters per size or use a parameterisation that transfers them (Chapter 7), and confirm with one run at the largest affordable scale before committing the full budget (Chapter 13.4).

### Triage training runs like experiments

A loss spike or a divergence is a research question with a cheap first experiment, not a reason to restart with a lower learning rate. The triage that usually works: look at the batch that caused the spike (a corrupted shard, a run of repeated documents, an extreme sequence length); look at the gradient norm and the largest logits or attention scores before it (logit growth points to the fixes of Chapter 7, such as QK-norm or a z-loss); then rewind to a checkpoint before the spike and replay with that batch skipped. If the spike does not recur, it was the data; if it recurs at a different step, it is the optimisation, and the learning rate, warm-up or precision is the next suspect (Chapter 17.9). Write each incident in the run log: a team that keeps these notes debugs its next run in hours instead of days.

### Correct for multiple comparisons

An ablation grid makes many comparisons at once, and some will look significant by chance. With 20 independent comparisons at p < 0.05 and no real effects, the probability that at least one looks significant is 1 − 0.95²⁰ = 64%. Three remedies, from simplest to most useful: a Bonferroni threshold (0.05 / 20 = 0.0025 per comparison), which is conservative; the Holm procedure, which is uniformly more powerful with the same guarantee; and controlling the false discovery rate (Benjamini–Hochberg) when the grid is exploratory and a few false leads are acceptable. Whatever the method, the habit that matters is reporting how many comparisons were made, so a reader can judge the one you highlight.

## Pitfalls

- Starting experiments before writing down what result would change your mind.
- Comparing a tuned method with an untuned baseline.
- One seed per configuration; reporting the best of a sweep without re-running it.
- One-at-a-time ablations for interacting components.
- Scaling up an idea that has been tested at one small scale.
- Not logging code and data versions, then being unable to reproduce your own best run.
- Writing only at the end.
- Planning a whole study around a noise estimate from three runs.
- Comparing small models on benchmarks where both sit at chance.
- Highlighting the one significant cell of a 20-cell ablation grid without saying how many cells there were.

## Interview questions

**1. How do you decide what to work on?** Impact, tractability, my edge and timeliness; a one-page proposal with a falsifiable hypothesis and the cheapest experiment that could kill it; time-boxed exploration with explicit continue/pivot/kill decisions. Impact is a product of how much the result matters and how likely it is to hold: an exciting optimiser claim with a low survival rate can be worth less than a modest data result the team will act on. When not to follow the framework: when a direction is cheap to try and the downside is a day, just run the experiment. Follow-up: "give an example of something you killed." Name the kill criterion and the experiment that triggered it (34.2's worked choice).

**2. A paper claims a 2-point gain. How do you assess it?** First, is 2 points larger than the noise? On a 500-question benchmark at 70% accuracy, one model's score has a standard error of √(0.7 × 0.3 / 500) = 2.0 points, so the difference between two independently evaluated models has a standard error near 2.9 points, and a 2-point gain is not distinguishable from noise unless the paper reports a paired comparison, several seeds or a larger set. Then check whether baselines were re-run and tuned with the same budget (a per-method learning-rate sweep), compute and data matched, results consistent across scales and tasks, the evaluation sound (contamination, judges), and whether the reported number is the best of several runs. Then try to reproduce the smallest setting. Follow-up: "what if they report 5 seeds?" Then ask for the interval and whether the seeds were paired across methods.

**3. Your change improves the score by 0.3 points on one seed. What next?** Estimate seed variance on the baseline first: if the seed SD is 0.4, a single comparison favours the new run 70% of the time even with this real gain, and 50% of the time with no gain. To detect 0.3 with 80% power at p < 0.05 needs n = 2(1.645 + 0.84)² × 0.4² / 0.3² ≈ 22 seeds per arm (23 with the exact t-test), so either reduce the noise (continuous metric, larger or paired evaluation) or test at a scale where the effect is larger. Do not claim anything until the interval excludes zero. When not to run more seeds: when even the optimistic end of the interval is too small to change the decision. Follow-up: "what if seeds are expensive?" Pair the seeds across arms and analyse differences.

**4. Why do one-at-a-time ablations mislead?** They miss interactions: a component that helps only in combination with another looks useless when added alone. In 34.5, B adds nothing to the baseline (−0.25 on one run, +0.03 in the factorial) but a full point when A is present. Use leave-one-out from the full system or factorial designs; with k components a full factorial costs 2^k cells, so use it for two to four components and leave-one-out beyond that. Follow-up: "how do you report interactions?" A 2 × 2 table of cell means with intervals.

**5. What is the winner's curse in hyperparameter sweeps?** The best observed configuration's score is biased upwards by noise, and re-running it gives a lower number. Its size is the expected maximum of k noise draws: about 1.63σ for 12 equally good configurations, so with σ = 0.5 the winner looks 0.81 points better than it is; for large k it grows like σ√(2 ln k). Confirm with fresh seeds and report the whole sweep. Follow-up: "does it matter if configurations really differ?" The bias shrinks but stays positive, so the fix is the same.

**6. How would you budget compute for testing a new idea?** A small-scale ladder with a few learning rates and seeds establishes the effect and its trend; a confirmation at the target scale follows only if the trend holds. Concretely, in 34.7: four sizes from 125M to 1.4B at 20 tokens per parameter, three learning rates and two seeds each, cost 1.9 × 10²¹ FLOPs or about 1,350 GPU-hours at 40% MFU on an H100, against 4,130 for one 7B confirmation run, so the ladder is 25% of the budget. When not to scale up: when the effect shrinks across the ladder, or when even the optimistic extrapolation is too small to change a decision. Follow-up: "how confident is the extrapolation?" Bootstrap the fit and report the interval (34.12).

**7. How do you make your experiments reproducible?** Config files, tracked code and data versions, seeds, automatic logging of all metrics and artefacts, versioned evaluation code, and documented variance where bit-exact determinism is impractical. A test that the setup works: two runs with the same seed give identical numbers (the check script's toy does), and the seed-to-seed spread is measured and reported. Follow-up: "what if the GPU kernels are non-deterministic?" Accept it, measure the variance it adds, and size the seed count accordingly.

**8. Describe a research result you got wrong, and what you changed afterwards.** (Behavioural, see Chapter 38.) Strong answers name a specific methodological lesson: added seeds, re-tuned baselines, changed evaluation. A typical strong story: a 0.4-point gain from one seed that vanished at five seeds, after which the speaker measured the noise floor before every study and pre-registered the seed count. Follow-up: "how did you find out?" Say who noticed and what check revealed it.

**9. Your 125M model scores at chance on the benchmark you care about. How do you run the ablation?** Use a metric that moves at this scale: held-out loss or bits per byte on domain text, or the probability assigned to the correct answer in a cloze format. At chance, a 4-option benchmark with 1,000 items has a standard error of 1.4 points, so 26% versus 27% is noise. Validate the proxy where the benchmark is above chance, and check that the effect keeps its sign across two or three ladder sizes before trusting the extrapolation. Follow-up: "what if the proxy and the benchmark disagree at the largest size?" Trust the benchmark there, and report the disagreement as a limit of the proxy.

**10. You ran 20 ablations and one is significant at p = 0.03. Do you believe it?** Not yet: with 20 comparisons and no real effects, the chance of at least one p < 0.05 is 1 − 0.95²⁰ = 64%, and a Bonferroni threshold would be 0.0025. Re-run that comparison with fresh seeds as a confirmation, and report all 20 alongside it. When a correction is too conservative for an exploratory grid, control the false discovery rate instead. Follow-up: "what if the effect is large?" A large effect with a modest p-value is usually a noisy estimate; the confirmation run will say.

## Practice

1. Pick a 2024–2026 paper close to your interests. Write the claims-and-doubts note from 34.3 and identify the smallest experiment that would test its main claim.
2. Reproduce one figure from a paper at small scale; document every discrepancy and how you resolved it.
3. Run 5 seeds of a small fine-tuning task; compute the seed standard deviation and the number of seeds needed to detect a 1-point gain with 80% power.
4. **Research exercise.** Take a published "improvement" to a small training recipe (an optimiser tweak, a regularisation trick). Re-run it against a baseline tuned with the same budget, 5 seeds each, at two model sizes. Write a research memo with the template in 34.9, whatever the outcome.
5. Write a one-page proposal for a project you would like to do, with the kill criterion stated.
6. Simulate the winner's curse: draw sweeps of k equally good configurations for k = 2, 12 and 100 and compare the mean of the maximum with the table in 34.6.

<details><summary>Hint and expected result</summary>

Item 1: a good note fits on half a page and names one table or figure that carries the main claim. Item 2: expect at least one unreported detail (warm-up, a data filter, an evaluation prompt) to explain most of the gap; bisect by swapping components one at a time. Item 3: use n = 2(1.645 + 0.84)²σ²/δ². With a seed SD of 0.5 points and δ = 1, n ≈ 3.1, so 4 seeds per arm (the exact t-test reaches 80% power at 4); with an SD of 1.5 points, n ≈ 28. Five seeds give a very uncertain SD (the 95% interval is roughly 0.6 to 2.9 times the estimate with 4 degrees of freedom), so state the range of n it implies. Item 4: the most common outcome is that the gain shrinks or disappears against a tuned baseline, or holds at one size only; a clear memo of that result is a success. Item 5: the kill criterion should be a number ("stop if the 125M difference is below one seed SD"). Item 6: about 0.56σ, 1.63σ and 2.51σ, matching the table to two decimals with 10,000 sweeps.

</details>

## Read next

- Richard Hamming, "You and Your Research" (talk, 1986).
- John Schulman, "An Opinionated Guide to ML Research" (2020).
- Andrej Karpathy, "A Recipe for Training Neural Networks" (2019); Google Research, "Deep Learning Tuning Playbook" (2023).
- Melis, Dyer and Blunsom, "On the State of the Art of Evaluation in Neural Language Models" (2017).
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021); Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (2021).
- S. Keshav, "How to Read a Paper" (2007).
- Links are in the Key concepts section at the top of this chapter.

## Done when

- Without notes, you derive n = 2(z_α + z_β)²σ²/δ² in five minutes and compute seeds per arm for σ = 0.4, δ = 0.3 (≈ 22; 23 with the t-test).
- You compute the chance that one seed of a real 0.3-point gain looks better at σ = 0.4 (Φ(0.53) = 0.70).
- You estimate the winner's curse for a sweep of k equal configurations from the table or from σ√(2 ln k), and get 0.81 for k = 12, σ = 0.5.
- You compute the GPU-hours of a 6ND budget at a given MFU in two minutes: the 34.7 ladder is about 1,350 H100-hours at 40% MFU.
- You state the 95% interval for σ estimated from 3 seeds (0.52 to 6.3 times the estimate) and what it means for planning.
- You explain, with the 2 × 2 table, why one-at-a-time ablations miss interactions.
- You compute the chance of a false positive among 20 comparisons (64%) and the Bonferroni threshold (0.0025).
- You have written one research proposal with a numeric kill criterion and one research memo in the 34.9 format.

---

[← 33 · Agents and tool use](33-agents-and-tool-use.md) · [Contents](../README.md#contents) · [35 · The ML coding interview →](35-the-ml-coding-interview.md)
