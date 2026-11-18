# 24 · LLM evaluation

Part VII · Evaluation science. Builds on Chapter 4 (metrics and confidence intervals) and Chapter 22 (prompt iteration). Chapter 25 applies this to RAG and agents. Worked design that builds on it: E.6 LLM judge for recommendation relevance.

## Why it matters

Evaluation is the skill that separates LLM demos from LLM products. Without it, every prompt change is a guess, every model upgrade is a risk, and nobody can say whether the feature is getting better. With it, you can iterate quickly and safely, and defend decisions with numbers. Practitioners who build LLM products for a living consistently say the evaluation system is the most important thing they build.

For your interviews, this is a differentiator: many candidates can build a RAG demo, few can explain how they would know it works. LLM judges, calibration against humans and sample sizes come up in both the ML depth and design rounds (worked design E.6).

**Where this chapter fits.** This chapter is about evaluating LLM *applications*: building an evaluation set for a product, task metrics, judges, and regression testing for prompts and models. Evaluating *models* as a research scientist (benchmark statistics, clustering, contamination, arenas, IRT) is covered in Chapter 23, and the two share the same statistical core.

## Key concepts

- **Benchmarks vs application evaluations**: public benchmarks compare models in general; your own evaluation set tells you whether your feature works. [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/).
- **Golden set**: a fixed, versioned set of real inputs with expected outputs or properties, run on every change.
- **Reference-based vs reference-free metrics**: comparing with a known answer (exact match, F1, ROUGE) vs judging the output alone (faithfulness, helpfulness).
- **LLM-as-judge**: using a strong model with a rubric to grade outputs; it must be validated against human labels and checked for position and length bias. [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685).
- **Holistic evaluation**: measuring many properties (accuracy, calibration, robustness, bias, cost), not one score. [HELM](https://arxiv.org/abs/2211.09110).
- **Contamination**: benchmark test data leaked into training data, inflating scores.
- **Online evaluation**: user feedback, A/B tests and sampled judging of live traffic (Chapter 47).

## 24.1 Benchmarks vs application evaluations

**Benchmarks** (MMLU, GPQA, SWE-bench, MTEB, and many more) compare models on fixed public tasks. They help shortlist models, but they do not tell you how a model performs on *your* task: your inputs, your definitions of good, your failure costs. Public benchmarks also leak into training data and saturate quickly.

**Application evaluations** measure your system on your task: your prompts, your retrieval, your users' questions, scored against your criteria. They are what you ship on.

Three levels of application evaluation, from cheapest to most expensive:

| Level | What | When it runs |
|---|---|---|
| Unit-style checks | Assertions in code: valid JSON, length, required fields, no PII, citations present | Every change, in CI, and on production outputs |
| Model and human evaluation | LLM judges calibrated against human labels on an evaluation set; periodic human review | Every prompt or model change; weekly samples of production |
| Online evaluation | A/B tests and user signals on real traffic | Before and after launches |

## 24.2 Error analysis is the starting point

The most common mistake is starting from a list of generic metrics ("helpfulness, coherence, relevance") and building judges for them. Generic metrics measure things that may not be your problem and miss the things that are. Start from failures instead:

![Diagram](images/22-01.png)

*the error-analysis loop · six steps, repeated*

1. **Collect traces:** real or realistic inputs with the full system output and intermediate steps (retrieved documents, tool calls).
2. **Read and take notes:** one person who understands the domain reads 50–100 traces and writes a short free-form note on what went wrong in each ("cited the Spain policy for a UAE user", "ignored the 24-hour limit"). This step is often called open coding.
3. **Group into failure modes:** cluster the notes into categories and count them. Now you know your top three problems and how often they occur.
4. **Write an evaluation per failure mode:** a code check when the failure is objective (wrong country cited: check metadata), a calibrated LLM judge when it needs judgement.
5. **Fix the biggest mode,** then 6. **re-run everything** and check for regressions.

Repeat whenever the system or its inputs change. The categories you discover become your evaluation suite, and they are specific to your product, which is exactly why they work. Reading data is not a chore to delegate; it is the highest-value hour in LLM development.

## 24.3 Building an evaluation set

**Sources, best first:** real production inputs (sampled, anonymised); inputs from the people who will use the system; failures found in error analysis; hand-written edge cases; synthetic inputs generated by an LLM to fill gaps (vary them by persona, intent, difficulty and language, and review them, because generated inputs are cleaner and more uniform than real ones).

**Size.** Start with 50–100 cases you understand deeply; grow to a few hundred as failure modes emerge. Size follows from the precision you need (24.8): 100 cases give a pass rate within about ±7 percentage points near 85% and ±10 near 50% (95% interval); detecting a 5-point improvement reliably needs many hundreds.

**Stratify** by the dimensions that matter: intent, language, difficulty, user type, and the failure modes you found. Report results per stratum; an average hides a broken slice.

**What to store per case:** the input, any context it needs (user profile, documents), the expected output or the criteria a good output must meet, tags for strata, and the source. For tasks with one right answer, store it; for open-ended tasks, store what a good answer must contain and must not contain.

**Keep it alive.** Version the set; add every production failure worth fixing; retire cases that no longer reflect the product; keep a held-out slice you never tune prompts on.

## 24.4 Reference-based metrics

When there is a known correct answer:

- **Exact match** after normalisation (lower case, strip punctuation and articles): for short factual answers, labels, ids.
- **Token F1:** word overlap between prediction and reference, the standard for extractive question answering.
- **BLEU, ROUGE, BERTScore:** overlap metrics for longer text; weak for LLM outputs (Chapter 41.9 shows a reversed meaning scoring higher).
- **Task metrics:** accuracy or F1 for classification, execution accuracy for generated SQL or code (run it and compare results), JSON field accuracy for extraction.

```python
import re, string
from collections import Counter

def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))

def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)

exact_match("The 24 hours.", "24 hours")             # 1.0
token_f1("within 24 hours of delivery", "24 hours")  # 0.57
```

## 24.5 Code-based checks

Many failure modes are objective and cheap to test in code. Write these first: they are fast, free, deterministic, and run on every production output, not just the evaluation set.

```python
import json, re

def run_checks(output, checks):
    results = {}
    for name, fn in checks.items():
        try:
            results[name] = bool(fn(output))
        except Exception:
            results[name] = False          # a crashing check is a failed check
    return results

checks = {
    "valid_json": lambda o: isinstance(json.loads(o), dict),
    "has_answer": lambda o: len(json.loads(o)["answer"]) > 0,
    "under_80_words": lambda o: len(json.loads(o)["answer"].split()) <= 80,
    "cites_source": lambda o: len(json.loads(o)["sources"]) > 0,
    "no_email": lambda o: re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", o) is None,
}
```

Other useful checks: the cited ids exist in the retrieved context; the answer is in the user's language; the classification label is in the taxonomy; a refund amount does not exceed the policy cap; the generated SQL parses and only touches allowed tables.

## 24.6 LLM-as-judge

When quality needs judgement (is the answer faithful? does it address the question? is the tone right?), an LLM can score outputs at scale. Three formats:

| Format | How | Good for | Watch out |
|---|---|---|---|
| Pointwise | Score one output against criteria | Monitoring, absolute pass rates | Scores drift; scales (1–10) are poorly calibrated |
| Pairwise | Which of two outputs is better? | Comparing prompts or models | Position bias; no absolute level |
| Reference-guided | Compare the output with a reference answer or required facts | Tasks with known answers but free wording | Needs references |

**How to write a judge that works:**

- **One judge per failure mode,** not one judge for "quality". "Does the answer state a refund amount not present in the documents?" is measurable; "is it good?" is not.
- **Binary pass/fail with a short critique** beats a 1–10 score: easier to calibrate, easier to act on, and the critique makes errors debuggable. Ask for the critique first, then the verdict.
- **Give the judge what it needs:** the input, the output, the retrieved context for faithfulness, the reference if any, and precise criteria with examples of pass and fail.
- **Use a strong model as judge,** and keep the judge prompt versioned like any other prompt.

```text
You are checking an answer from a customer-support assistant.
Criterion: every factual claim in the ANSWER must be supported by the DOCUMENTS.
A claim is unsupported if it adds amounts, deadlines or conditions not in the documents.

DOCUMENTS: {documents}
QUESTION: {question}
ANSWER: {answer}

First write a short critique listing any unsupported claims.
Then output JSON: {"critique": str, "verdict": "pass" | "fail"}
```

## 24.7 Judge biases and calibration against humans

LLM judges have known biases (Zheng et al., 2023, and later work):

- **Position bias:** in pairwise comparisons, preferring the first (or second) answer. Fix: run both orders and only count consistent verdicts.
- **Verbosity bias:** preferring longer answers. Fix: criteria that penalise padding; length-controlled comparisons.
- **Self-preference:** preferring outputs from the same model family. Fix: a judge from a different family, or calibration.
- **Leniency or harshness** that differs by slice or language.

**Calibrate every judge before trusting it.** Have a domain expert label 100–200 outputs for the same criterion, run the judge on them, and measure agreement:

```python
import numpy as np

def cohens_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)                                       # observed agreement
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in np.union1d(a, b))   # chance agreement
    return (po - pe) / (1 - pe)

def judge_tpr_tnr(human, judge):
    h, j = np.asarray(human), np.asarray(judge)
    return np.mean(j[h == 1] == 1), np.mean(j[h == 0] == 0)   # pass-recall, fail-recall
```

**Read agreement correctly.** Raw agreement looks high when most outputs pass. Cohen's kappa corrects for chance (rough guide: above 0.6 is substantial, above 0.8 near-perfect). Even more useful: the judge's **true positive rate** (catches real passes) and **true negative rate** (catches real failures). A judge that rarely flags failures is useless for finding bugs even if agreement is 90%. Compare judge-human agreement with human-human agreement on a sample: if two experts agree only 85% of the time, do not expect 95% from a judge.

**Correct the pass rate for judge errors.** If the judge's TPR and TNR are known, an observed pass rate can be corrected (the Rogan–Gladen estimator from epidemiology):

```python
def corrected_pass_rate(observed_rate, tpr, tnr):
    return min(max((observed_rate + tnr - 1) / (tpr + tnr - 1), 0.0), 1.0)

# Judge with TPR 0.93, TNR 0.67 reports 80% pass; the corrected estimate is about 78%
```

Iterate on the judge prompt until agreement is good enough, using the human labels as its own evaluation set, with a held-out part.

## 24.8 Sample sizes and confidence intervals

Every evaluation number is an estimate. Report intervals, and compare systems with paired tests.

```python
import math
from scipy import stats

def wilson_interval(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half

wilson_interval(85, 100)     # (0.77, 0.91): "85%" on 100 cases is anything from 77% to 91%
wilson_interval(850, 1000)   # (0.83, 0.87)

def mcnemar_exact(a_correct, b_correct):
    """Paired comparison of two systems on the same cases: only disagreements matter."""
    n01 = int(sum(a and not b for a, b in zip(a_correct, b_correct)))
    n10 = int(sum(b and not a for a, b in zip(a_correct, b_correct)))
    n = n01 + n10
    return n01, n10, (stats.binomtest(min(n01, n10), n, 0.5).pvalue if n else 1.0)
```

**Rules of thumb.** To detect an improvement from 80% to 85% with two independent samples, you need roughly 900 cases per arm; from 80% to 82%, about 6,000. Paired evaluation on the same cases (McNemar) needs far fewer, because only the cases where the systems disagree carry information. So: always compare prompts and models on the **same** cases, and do not celebrate a 2-point gain on 200 examples.

## 24.9 Online evaluation

Offline sets never fully match production. Online signals:

- **Explicit feedback:** thumbs up or down, ratings. Sparse and biased towards unhappy users, but valuable for finding failures.
- **Implicit feedback:** the user copies the answer, edits the draft (and how much), asks the same question again, escalates to a human, abandons the session.
- **Business outcomes:** contact deflection, resolution without escalation, conversion, time saved.
- **Sampled judging:** run the calibrated judges on a sample of production traffic daily and alert on drops.

**A/B testing LLM features** follows Chapter 2, with extras: guardrails on cost per request and latency; novelty effects (users try the new assistant more at first); and interference when the feature changes what users ask. Ship behind flags, start with a small share of traffic, and compare both the user metrics and the judged quality.

## 24.10 Tooling

You can do all of this with Python, a spreadsheet and git. Tools help at scale: tracing and observability (Langfuse, LangSmith, Arize Phoenix, Braintrust, Weights & Biases Weave) store traces, run evaluations, manage datasets and judges, and show results per version. Pick one that stores full traces and lets you annotate them, because annotation is where error analysis happens. Avoid tools that push you to generic out-of-the-box metrics before you have read your data.

## Pitfalls

- **Generic metrics first.** "Helpfulness" scores that do not match your real failure modes.
- **Never reading the data.** Dashboards of judge scores without anyone reading traces.
- **Uncalibrated judges.** A judge nobody checked against human labels is an opinion, not a metric.
- **Reporting agreement without kappa, TPR and TNR.** A judge can agree 90% and still miss most failures.
- **1–10 scores.** Hard to calibrate, easy to drift; prefer binary verdicts per criterion.
- **Comparing systems on different samples** or without intervals.
- **Pairwise judging in one order only** (position bias).
- **Tuning prompts on the whole evaluation set,** then reporting on it.
- **Evaluation sets that never change** while the product and users do.
- **Synthetic-only evaluation sets** that are cleaner than real inputs.
- **Ignoring cost and latency** in the evaluation table.

## Interview questions

Questions 1–4 basic, 5–9 intermediate, 10–13 senior.

**1. Why are public benchmarks not enough to choose a model for your product?** They test generic tasks, may be contaminated, and ignore your inputs, criteria and failure costs. Use them to shortlist, then evaluate candidates on your own set.

**2. How do you start evaluating a new LLM feature?** Error analysis: collect 50–100 realistic traces, read and annotate failures, group them into failure modes, count them, then build code checks or calibrated judges for the most frequent modes.

**3. What is LLM-as-judge, and its main formats?** Using an LLM to grade outputs against criteria: pointwise (one output), pairwise (which of two), reference-guided (against a known answer). Best as binary verdicts per criterion with a critique.

**4. Exact match vs token F1?** Exact match requires the normalised strings to be identical; token F1 gives partial credit for word overlap. Both suit short answers with a reference; neither suits open-ended generation.

**5. What biases do LLM judges have and how do you handle them?** Position, verbosity, self-preference, and slice-dependent leniency. Swap orders in pairwise judging, penalise padding, use a different model family, and calibrate against human labels per slice.

**6. How do you validate an LLM judge?** Expert labels on 100–200 outputs for the same criterion; measure kappa and the judge's TPR and TNR; compare with human-human agreement; iterate on the judge prompt; keep a held-out set; re-check after judge or model changes.

**7. Your judge agrees with humans 92% of the time. Is it good?** Not necessarily: if 90% of outputs pass, a judge that always says pass agrees 90%. Check kappa and especially the TNR: the share of real failures it catches.

**8. How many examples do you need in an evaluation set?** Enough for the precision you need: 100 cases gives roughly ±7 points on a pass rate near 85% (±10 near 50%); detecting small differences between systems needs paired comparisons on the same cases or many hundreds of examples.

**9. How do you compare two prompts fairly?** Same cases, same judge, paired test (McNemar on per-case pass/fail), intervals, per-slice results, plus cost and latency.

**10. Design the evaluation for a customer-support RAG assistant.** Error analysis on real tickets; code checks (format, citations exist, language, no PII, policy caps); calibrated judges for faithfulness, answer correctness against references for common questions, and correct abstention; retrieval metrics separately (Chapter 25); per-country and per-language slices; production sampling with daily judging; online metrics (resolution without escalation, re-contact rate, CSAT) with cost and latency guardrails.

**11. Judge scores went down 5 points after a model upgrade of the judge itself. What happened?** The measuring instrument changed: the new judge may be stricter or differently biased. Re-calibrate against the human labels before interpreting any trend; pin judge versions and change them deliberately.

**12. How do you evaluate when there is no single right answer (for example writing restaurant descriptions)?** Define concrete criteria from error analysis (factual consistency with menu data, no invented claims, length, tone, banned phrases), code checks where possible, calibrated binary judges for the rest, pairwise preference tests between versions, and human review samples.

**13. How would you convince a product manager that an LLM feature is ready to launch?** Show per-failure-mode pass rates with intervals on a representative set, critical slices, judge calibration evidence, cost and latency, a comparison with the current process, the monitoring plan, and the rollout plan with guardrails and rollback criteria.

## Practice

1. **Error analysis.** Take the RAG system from Chapter 31's practice. Read 60 traces, write notes, group them into failure modes, count them.
2. **Code checks.** Write `run_checks` checks for three of the modes you found and run them on all outputs.
3. **A judge with calibration.** Write a faithfulness judge; label 100 outputs yourself; compute kappa, TPR and TNR; improve the judge prompt twice and report the change.
4. **Pairwise judging.** Compare two prompt versions with a pairwise judge in both orders; count inconsistent verdicts.
5. **Statistics.** Compute Wilson intervals for each failure mode's rate and a McNemar test between the two prompts.
6. **Reading.** Write a one-page evaluation plan for worked design E.6 before re-reading the model answer.

## Read next

- Hamel Husain's posts *Your AI Product Needs Evals* and *Creating a LLM-as-a-Judge* *That Drives Business Results*, and Husain and Shankar's material on evaluation (hamel.dev).
- Shreya Shankar et al., *Who Validates the Validators? Aligning LLM-Assisted Evaluation* *of LLM Outputs with Human Preferences* (2024).
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023).
- *AI Engineering* (Chip Huyen): the chapters on evaluation methodology and evaluating AI systems.
- Eugene Yan's writing on evals and LLM judges (eugeneyan.com).
- Anthropic and OpenAI documentation on building evaluations.

**Links:** [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) · [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) · [HELM](https://arxiv.org/abs/2211.09110). Other titles above can be found by searching their title.

## Done when

You can explain benchmarks vs application evals and the three levels of evaluation. You can run an error analysis and turn failure modes into evaluations. You can design an evaluation set: sources, size, strata, maintenance. You can write exact match, token F1 and code checks from scratch. You can write a binary judge with critique and calibrate it with kappa, TPR and TNR.

You can name judge biases and their fixes. You can compute Wilson intervals and run a paired comparison. You can design online evaluation for an LLM feature.

---

[← 21 · Agents and tool use](21-agents-and-tool-use.md) · [Contents](../README.md#contents) · [23 · RAG and agent evaluation →](23-rag-and-agent-evaluation.md)
