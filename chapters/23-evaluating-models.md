# 23 · Evaluating models

Part VII · Evaluation science. Assumes Chapter 2 (confidence intervals, hypothesis tests) and Chapter 4 (metrics). Chapter 24 covers evaluating LLM *applications* (task metrics, judges, eval sets for a product); Chapter 25 covers RAG and agents. This chapter is about evaluating *models*: benchmarks, statistics, contamination and comparisons.

## Why it matters

Every decision in model development rests on an evaluation: which data mixture to keep, which checkpoint to release, whether an architecture change helped, whether a paper's claim is real. Yet benchmark scores are noisy, often contaminated, sensitive to prompt formats, quickly saturated and sometimes measuring something other than their name. A research scientist is expected to treat evaluation as measurement science: quantify uncertainty, separate signal from noise, detect contamination, and know what a benchmark actually measures. Interviewers test this with questions like "model B beats A by 1.5 points on MMLU; is it better?", and the right answer starts with a standard error.

Numbers in this chapter come from `code/ch23_check.py` (seconds).

## Key concepts

- **Error bars for evals**: standard errors, clustering and paired comparisons. [Miller, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640).
- **Benchmark design and saturation**: harder, expert-written, sometimes held-out benchmarks. [GPQA](https://arxiv.org/abs/2311.12022), [Humanity's Last Exam](https://arxiv.org/abs/2501.14249).
- **Item response theory** for efficient and informative evaluation. [tinyBenchmarks](https://arxiv.org/abs/2402.14992).
- **Contamination** and how to detect it. [Min-K% Prob](https://arxiv.org/abs/2310.16789), [LiveCodeBench](https://arxiv.org/abs/2403.07974).
- **Pairwise comparison and arenas**. [Chatbot Arena](https://arxiv.org/abs/2403.04132), [The Leaderboard Illusion](https://arxiv.org/abs/2504.20879).
- **LLM-as-a-judge biases**. [Judging LLM-as-a-judge](https://arxiv.org/abs/2306.05685), [Length-controlled AlpacaEval](https://arxiv.org/abs/2404.04475).
- **Long-horizon capability measurement**. [METR time horizons](https://arxiv.org/abs/2503.14499).
- **Harnesses**: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [Inspect](https://inspect.aisi.org.uk/), [HELM](https://arxiv.org/abs/2211.09110).

## 23.1 What evaluation is for

An evaluation exists to support a decision. Before choosing a metric, write down the decision ("ship checkpoint 41 or 38?", "keep the new data filter?") and the smallest difference that would change it. That fixes how precise the evaluation must be, and therefore how large.

Four levels, each with its own role:

| Level | Examples | Good for | Weak at |
|---|---|---|---|
| Loss on held-out data | Per-domain perplexity, bits per byte | Smooth, cheap, comparable across checkpoints and small scales | Not a capability measure; tokenizer-dependent |
| Capability benchmarks | MMLU-Pro, GPQA, MATH, HumanEval, SWE-bench | Standardised comparison | Saturation, contamination, format sensitivity |
| Preference and arena comparisons | Pairwise human or judge votes | Open-ended quality | Style and length biases; selection effects |
| Task-specific and human evaluations | Your product's eval set (Chapter 24) | What you actually care about | Expensive; small |

**Construct validity**: does the benchmark measure the ability its name claims? A "reasoning" benchmark answerable by pattern-matching, a "long-context" benchmark solvable from the last page, a "coding" benchmark of short puzzles: all common. Read 20 items of any benchmark before trusting it.

## 23.2 Error bars

**The standard error of a mean.** A benchmark score is an average over questions, which are a sample from some population of possible questions. With n independent questions and accuracy p, the standard error is √(p(1−p)/n): about 2.2 points for 500 questions at 67%, so a 95% interval of ±4.3 points.

**Clustered questions.** Many benchmarks group questions (several questions per reading passage, per code repository, per document). Questions in a cluster are correlated, so the effective sample size is smaller. In the check script, 500 questions in 100 passages of 5 give a naive standard error of 0.0211 and a clustered one of 0.0261: 24% larger. The clustered formula sums residuals within each cluster before squaring.

**Paired comparisons.** Comparing two models on the same questions, analyse the per-question differences, not two separate means: question difficulty is shared, and pairing removes it. In the check script, model B is truly about 2.5 points better than A on average. On 500 questions the observed difference is −0.6 points, with a paired 95% interval of [−5.7, +4.5]: 500 questions cannot detect a gain of this size. The paired standard error (0.026) is smaller than the unpaired (0.031), and the gap grows with how much the models agree.

**Sample size.** For a paired comparison, the number of questions needed to detect a difference δ with 80% power is about (1.96 + 0.84)² × d / δ², where d is the fraction of questions on which the models disagree. Detecting a 2-point gain needs about 2,000 questions if they disagree on 10%, and about 4,000 at 20%. Many benchmark comparisons in papers are underpowered.

**Many comparisons.** Comparing 20 checkpoints and picking the best is a multiple-comparison problem; the winner's score is biased upwards. Confirm on a held-out set or correct for multiplicity (Chapter 2).

**Report**: the mean, its standard error (clustered if relevant), the number of questions, and for comparisons the paired difference with its interval.

## 23.3 Other sources of variance

Sampling questions is only one source of noise:

- **Prompt format**: the same model scored 60.5% to 66.0% across six reasonable templates in the check script's simulation, a 5.4-point range; real studies find similar or larger spreads, especially for small models and multiple-choice formats. Fix the format and report it, or average over several.
- **Few-shot examples**: which examples, and in which order.
- **Decoding**: greedy versus sampling, temperature, maximum length (truncated reasoning counts as wrong).
- **Answer extraction**: regex differences can move scores by points.
- **Training seed**: two runs of the same recipe differ. For small models, seed variance on some benchmarks rivals the effect of the change being tested. Train 2–3 seeds of the baseline at least once to know the noise floor.
- **Harness and version**: the same benchmark in two harnesses can differ by several points.

## 23.4 Loss, perplexity and bits per byte

Held-out loss is the most sensitive and least noisy signal during pretraining, and it moves at scales where benchmarks are flat. But loss per token depends on the tokenizer: a tokenizer with longer tokens has fewer, harder predictions. Compare models with different tokenizers in **bits per byte**: total loss in nats over the text, divided by the number of bytes, divided by ln 2.

| Model | Loss (nats/token) | Perplexity | Bits per byte |
|---|---|---|---|
| A, 32k vocabulary, 1.00M tokens for 4.1 MB | 2.10 | 8.2 | 0.739 |
| B, 128k vocabulary, 0.82M tokens for 4.1 MB | 2.45 | 11.6 | 0.707 |

B looks worse per token but compresses the text better: it is the better model on this data.

**Log-likelihood versus generative evaluation.** Multiple-choice benchmarks can be scored by comparing the model's likelihood of each option ("cloze" or ranked classification) or by generating an answer letter. Likelihood scoring works for small base models that cannot yet follow the answer format; generation is what users see. Report which you used, and do not compare across them.

## 23.5 Benchmark design

Benchmarks have a life cycle: introduced, climbed, saturated, sometimes contaminated, retired. MMLU went from hard to saturated in about four years; harder successors (MMLU-Pro, GPQA's expert-written questions, Humanity's Last Exam) followed. Good benchmark design:

- **Clear construct** and items that require it (no shortcuts from answer formats or artefacts).
- **Verified labels**: popular benchmarks have measurable label error rates (several percent in some MMLU subjects), which caps achievable accuracy and adds noise at the top.
- **Difficulty spread** so it discriminates between the models you care about.
- **Held-out or refreshed items** to resist contamination (private test sets, items created after model cut-off dates).
- **Size** for the precision you need (23.2).

**Item response theory (IRT)** models each item's difficulty and discrimination and each model's ability, as in educational testing. In the 2PL model, the probability that model m answers item i correctly is σ(a_i(θ_m − b_i)). IRT is used to find informative subsets, detect bad items (negative discrimination usually means a label error) and compare models on a common scale. In the check script, with 30 simulated models and 1,000 items, a random 100-item subset correlates 0.968 with the full score; 100 informative items spread across difficulty correlate 0.979. tinyBenchmarks and similar work use this to cut evaluation cost by 10× or more with small loss of accuracy.

## 23.6 Contamination

If test items, or close paraphrases, were in the training data, scores measure memory. Contamination is the default assumption for any public benchmark older than a model's training cut-off.

**Detection when you have the training data**: n-gram overlap between test items and training documents (Chapter 16), plus embedding search for paraphrases.

**Detection when you only have the model**:

- **Perturbation gap**: rephrase items, or change numbers in maths problems, so that the answer must be recomputed. A large drop on perturbed versions, beyond what the change in difficulty explains, suggests memorisation. In the check script, accuracy falls from 75.2% to 67.5% on rephrased items; a McNemar test on the 51 lost and 20 gained items gives p = 0.0003.
- **Guided completion**: give the model the beginning of a test item and see whether it completes it verbatim.
- **Likelihood signals**: Min-K% Prob and similar tests compare the likelihood of the least likely tokens of an item with that of comparable unseen text. Useful for screening, unreliable as proof.
- **Time splits**: evaluate on items published after the model's cut-off (LiveCodeBench-style rolling benchmarks); a drop on post-cut-off items is strong evidence.

**Prevention**: canary strings in benchmark files (so they can be filtered), private held-out test sets, and refreshed benchmarks.

## 23.7 Pairwise comparisons and arenas

Open-ended quality is hard to score absolutely, so many evaluations compare responses in pairs: humans or judges vote, and a **Bradley–Terry** model (the same as for reward models, Chapter 18) turns votes into ratings, usually on an Elo-like scale. On 3,000 simulated battles among five models, fitted ratings recover the true order, but adjacent models' bootstrap intervals overlap (for example [1,181, 1,235] and [1,149, 1,205]): with this many votes, the top three are hard to separate.

Arenas with crowd-sourced prompts and votes have become the most cited leaderboards, and their known problems are worth knowing:

- **Style and length**: voters prefer longer, more formatted answers; style-controlled ratings regress the votes on length and markdown features.
- **Selection effects**: a developer who tests many private variants and publishes the best gets an inflated rating. If ten variants of equal strength are tested and the best published, its expected inflation here is about 21 rating points, purely from noise. The Leaderboard Illusion paper (2025) documented such practices and unequal sampling between providers.
- **Prompt distribution**: arena prompts are not your users' prompts.

Use arenas as one signal among several, with intervals, and run your own pairwise evaluations on your own prompt distribution.

## 23.8 LLM judges as measurement instruments

LLM judges are cheap and scalable, and they are instruments with biases that must be measured (Chapter 24 covers building them for applications):

- **Position bias**: preferring whichever answer is shown first (or second). In the check script, a judge with a 10-point first-position bias reports B's win rate as 45.1% or 64.9% depending on order; averaging both orders recovers 54.9% against a true 55%. Always swap and average.
- **Length and verbosity bias**: correct with length-controlled win rates (regress out the length difference).
- **Self-preference**: judges favour outputs of their own model family. Use judges from a different family, or several.
- **Weak judges on hard tasks**: a judge that cannot solve the problem cannot reliably judge the solution. Give reference answers or use verifiers.

Validate any judge against human labels on a sample, per category, and report agreement with an interval.

## 23.9 Measuring agents and long-horizon capability

Single-turn benchmarks miss what matters for agents: completing long, multi-step tasks reliably. Two ideas to know:

- **Reliability over attempts**: pass^k (succeeds on all k tries) rather than pass@k (succeeds at least once); agents that succeed 60% of the time per attempt succeed on all of 8 attempts less than 2% of the time (Chapter 25).
- **Time horizons**: METR measures, for each model, the length of tasks (in human expert time) it completes with 50% reliability. That horizon has grown roughly exponentially, doubling every several months over the last few years: a capability trend that is easier to interpret than benchmark points.

Agentic benchmarks (SWE-bench Verified for software engineering, OSWorld for computer use, τ-bench for tool use with simulated users) need careful environment control: the same model can differ by many points with a different scaffold, tool set or time limit. Report the full setup.

## 23.10 Infrastructure and reporting

Use a harness (lm-evaluation-harness, Inspect, HELM, or your lab's own) so prompts, extraction and scoring are versioned code. Record model revision, harness version, prompt template, few-shot examples, decoding settings, number of samples and seeds. A reporting checklist for any model comparison:

1. What decision does this evaluation support?
2. Scores with standard errors (clustered where relevant) and the number of items.
3. Paired differences with intervals for comparisons.
4. Prompt format and decoding settings; variance across formats if checked.
5. Contamination checks done and their results.
6. For judges: which judge, bias controls, agreement with humans.
7. Seeds or training-variance estimates for training-time comparisons.

## Pitfalls

- Declaring a winner from a difference smaller than two standard errors.
- Ignoring clustering and paired structure.
- Comparing per-token perplexity across tokenizers.
- Comparing scores from different harnesses, formats or extraction rules.
- Trusting public benchmark scores without contamination checks.
- Using an arena rating without its interval, or ignoring selection effects.
- Using an LLM judge without position swapping and length control.

## Interview questions

**1. Model B beats A by 1.5 points on a 1,000-question benchmark. Is B better?** Compute the paired difference and its standard error: with ~70% accuracy and moderate disagreement, the SE of the difference is around 1–1.5 points, so 1.5 points is not convincing. Check clustering, prompt-format and seed variance, and contamination.

**2. Why cluster standard errors, and how?** Questions that share a passage, repository or template are correlated; the naive SE understates uncertainty. Sum residuals within clusters, then compute the variance across clusters.

**3. How many questions do you need to detect a 2-point improvement?** For paired comparisons, about 7.84 × d / 0.02² where d is the disagreement rate: roughly 2,000 questions at d = 10%.

**4. Why use bits per byte?** Loss per token depends on the tokenizer; bits per byte normalises by text length and allows comparisons across tokenizers.

**5. How would you detect contamination in a closed model?** Perturbation gaps with significance tests, guided completion, likelihood-based tests, and time-split evaluations on post-cut-off items.

**6. What is IRT and how is it used in LLM evaluation?** Models item difficulty and discrimination and model ability; used to choose small informative subsets, find mislabelled items and place models on a common scale.

**7. What are the known problems with arena leaderboards?** Style and length biases, selection effects from testing private variants, unequal sampling, prompt distribution mismatch, overlapping intervals.

**8. How do you make an LLM judge trustworthy?** Swap positions and average, control for length, use a different model family or several judges, provide references, validate against human labels per category with intervals.

**9. Why might a benchmark lack construct validity?** Items solvable by shortcuts, answer-format artefacts, label errors, or measuring a narrow skill under a broad name. Read items, test blind or perturbed baselines.

**10. What is METR's time-horizon metric?** The length of tasks (in human expert time) a model completes with 50% success; it tracks agentic capability over time in interpretable units.

## Practice

1. Take a public benchmark with passage structure; compute naive and clustered standard errors for an open model.
2. Evaluate two open models on 1,000 multiple-choice questions; compute the paired difference with a bootstrap interval and the McNemar test.
3. Score one model with five prompt templates and with likelihood versus generation; report the spread.
4. **Research exercise.** Build a perturbed version of 200 maths word problems (change the numbers, recompute answers). Evaluate three open models of different release dates on original and perturbed sets. Test the gaps for significance and discuss what they say about contamination.
5. Fit a 2PL IRT model to a public matrix of per-item results for many models; inspect items with negative discrimination.

## Read next

- Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations" (2024).
- Liang et al., "Holistic Evaluation of Language Models" (HELM, 2022).
- Rein et al., "GPQA" (2023); Phan et al., "Humanity's Last Exam" (2025).
- Polo et al., "tinyBenchmarks: evaluating LLMs with fewer examples" (2024).
- Shi et al., "Detecting Pretraining Data from Large Language Models" (Min-K% Prob, 2023); Jain et al., "LiveCodeBench" (2024).
- Chiang et al., "Chatbot Arena" (2024); Singh et al., "The Leaderboard Illusion" (2025).
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023); Dubois et al., "Length-Controlled AlpacaEval" (2024).
- Kwa et al., "Measuring AI Ability to Complete Long Tasks" (METR, 2025).
- Links to these papers are in the Key concepts section at the top of this chapter.

## Done when

You can compute standard errors, clustered standard errors and paired differences, and size an evaluation for a target difference. You can list the sources of variance beyond question sampling and control them. You can explain bits per byte and when to use likelihood versus generative scoring.

You can critique a benchmark's construct validity and explain IRT. You can detect contamination with and without training data. You can fit Bradley–Terry ratings with intervals, explain arena pitfalls, and de-bias an LLM judge. You can write an evaluation report that a sceptical reviewer would accept.

---

[← 22 · Prompting and context engineering](22-prompting-and-context-engineering.md) · [Contents](../README.md#contents) · [24 · LLM evaluation →](24-llm-evaluation.md)
