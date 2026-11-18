# 2 · Statistics and experimentation

Part I · Foundations. Assumes basic probability (Chapter 1). Chapter 4 covers confidence intervals on model metrics; this chapter covers experiments on products. Worked designs and chapters that rely on it: every "online evaluation" section, Chapter 45 (marketplace experiments), E.1 and E.8.

## Why it matters

Every model you ship is judged by an experiment, and in a delivery marketplace experiments are hard: effects are small, metrics are noisy, users interact, and couriers are shared between test groups. Data scientists at delivery and marketplace companies spend a large share of their time on experiment design and analysis, and interviewers test it directly ("how many users do you need?", "the test is significant on day 3, do you stop?", "how do you test a new dispatch algorithm?"). Getting these answers right is one of the clearest signals of seniority.

## Key concepts

- **Sampling distribution and the central limit theorem**: an average computed from a sample varies from sample to sample; for large samples that variation is roughly normal, which is what makes confidence intervals and tests work. [Central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem); see it animated at [Seeing Theory](https://seeing-theory.brown.edu/).
- **Confidence interval**: a range that would contain the true value in 95% of repeated experiments. [Confidence interval](https://en.wikipedia.org/wiki/Confidence_interval).
- **Bootstrap**: estimate uncertainty by resampling your data with replacement many times. [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
- **Hypothesis test and p-value**: how surprising the data would be if there were no effect. [P-value](https://en.wikipedia.org/wiki/P-value).
- **Statistical power and sample size**: the chance of detecting a real effect of a given size; decides how long a test must run. [Power](https://en.wikipedia.org/wiki/Power_(statistics)), [sample size calculator](https://www.evanmiller.org/ab-testing/sample-size.html).
- **A/B test (online controlled experiment)**: randomly split users and compare outcomes to measure causal impact. [A/B testing](https://en.wikipedia.org/wiki/A/B_testing).
- **Peeking and sequential testing**: checking results repeatedly and stopping when significant inflates false positives. [How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html).
- **Multiple comparisons**: testing many metrics or segments guarantees some false wins. [Multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem).
- **Variance reduction (CUPED) and interference (switchbacks)**: using pre-experiment data to shrink noise, and randomising by time and area when users share couriers. Covered in this chapter; the reference book is [Trustworthy Online Controlled Experiments](https://experimentguide.com/).

## 2.1 Sampling, confidence intervals and the bootstrap

**The sampling distribution.** A metric computed on a sample (a conversion rate, a mean order value) varies from sample to sample. Its standard deviation across samples is the **standard error**: for a mean, σ/√n; for a proportion, √(p(1−p)/n). The central limit theorem says the sampling distribution of a mean is approximately normal for large n, whatever the data's distribution.

**A 95% confidence interval** is estimate ± 1.96 × standard error. Correct reading: if we repeated the experiment many times, 95% of intervals built this way would contain the true value. It is not "a 95% probability that the true value is in this interval" (a Bayesian credible interval says that).

**The bootstrap** gives intervals for any statistic (medians, ratios, percentiles, model metrics) without formulas: resample the data with replacement many times, recompute the statistic, take percentiles. Resample the independent unit (users, not orders, if one user has many orders; Chapter 4.9).

```python
import numpy as np
rng = np.random.default_rng(0)

order_values = rng.lognormal(3, 0.5, 2000)            # skewed, like real basket values
boots = [np.median(rng.choice(order_values, len(order_values))) for _ in range(2000)]
np.median(order_values), np.percentile(boots, [2.5, 97.5])   # ~20.3, (19.7, 20.9)
```

**Ratio metrics** (orders per session, revenue per order) where both parts vary by user need the **delta method** or a user-level bootstrap; treating each order as independent understates the variance.

## 2.2 Hypothesis tests and p-values

**The logic:** assume no effect (the null hypothesis); compute how surprising the observed difference would be under that assumption. The **p-value** is the probability of a result at least this extreme *if the null were true*.

What a p-value is not: the probability that the null is true, the probability the result is a fluke, or the size or importance of the effect. A tiny, useless effect can have p < 0.001 with enough data.

**Errors:** a type I error (false positive) happens with probability α, usually 0.05; a type II error (false negative) with probability β; **power** = 1 − β, usually targeted at 0.8.

**Common tests:**

| Data | Test |
|---|---|
| Two proportions (conversion) | Two-proportion z-test, or chi-square |
| Two means (order value) | Welch's t-test (does not assume equal variances) |
| Heavily skewed metrics | t-test still works for large n by the CLT; consider winsorising outliers, a log transform, or bootstrap / Mann-Whitney for medians |
| Paired data (same users before and after, two models on the same cases) | Paired t-test, McNemar for binary outcomes |
| Counts across categories, sample ratio checks | Chi-square |

```python
import math
from scipy import stats

def two_prop_ztest(conv_a, n_a, conv_b, n_b):
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))            # SE under the null
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)     # SE for the interval
    return z, p_value, (p_b - p_a - 1.96 * se_diff, p_b - p_a + 1.96 * se_diff)

two_prop_ztest(1000, 20000, 1080, 20000)
# 5.0% vs 5.4% conversion: z = 1.80, p = 0.07, 95% CI of the difference (-0.04, +0.84) points
```

An 8% relative lift that is not significant with 20,000 users per arm: the next section explains why, and how many users it would take.

## 2.3 Power and sample size

Before running a test, decide the **minimum detectable effect (MDE)**: the smallest effect worth detecting. Then compute the sample size that detects it with the desired power.

For two proportions, per arm: `n ≈ (z₁₋α/₂ + z₁₋β)² · [p₁(1−p₁) + p₂(1−p₂)] / (p₂` `− p₁)²`. For means: `n ≈ 2(z₁₋α/₂ + z₁₋β)² σ² / δ²`.

```python
def n_per_arm_proportion(p_base, mde_rel, alpha=0.05, power=0.8):
    p2 = p_base * (1 + mde_rel)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil((za + zb) ** 2 * (p_base * (1 - p_base) + p2 * (1 - p2)) / (p2 - p_base) ** 2)

def n_per_arm_mean(sd, mde_abs, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil(2 * (za + zb) ** 2 * sd ** 2 / mde_abs ** 2)

n_per_arm_proportion(0.05, 0.05)   # 5% base conversion, detect +5% relative: ~122,000 per arm
n_per_arm_proportion(0.05, 0.02)   # detect +2% relative: ~753,000 per arm
n_per_arm_mean(25.0, 0.5)          # basket value sd 25 euros, detect +0.50: ~39,000 per arm
```

The rule to remember: **halving the MDE quadruples the sample size.** This is why small improvements need large traffic, variance reduction (2.5) or more sensitive designs (interleaving, 2.8). Run for whole weeks to cover weekly cycles, and compute duration from traffic before launching, not after.

## 2.4 A/B test design

A checklist that covers what interviewers probe:

1. **Hypothesis and decision:** what change, what you expect, and what you will do with each outcome.
2. **Randomisation unit:** user, session, order, restaurant, city, or time slot. Use the unit at which the treatment acts and at which interference is contained. The analysis unit must match (or use clustered standard errors).
3. **Metrics:** one primary metric (sensitive and aligned with the goal), secondary metrics, **guardrails** (latency, cancellations, delivery time, revenue) and counter-metrics (Chapter 4.7). Written down before the test.
4. **Sample size and duration** from the MDE (2.3); at least one or two full weeks.
5. **Checks during the test:** sample ratio mismatch, guardrail breaches, data quality. No peeking at the primary metric to decide early (2.7).
6. **Analysis:** effect with confidence interval, segments defined in advance, multiple-testing awareness (2.6).
7. **Decision and documentation. Sample ratio mismatch (SRM):** if you planned a 50/50 split and got 50,500 vs 49,500 users, a chi-square test gives p ≈ 0.0016: the assignment or logging is broken (bots, redirects, crashes in one arm), and the results cannot be trusted until the cause is found. Check SRM on every test.

```python
def srm_p_value(n_a, n_b, expected_ratio=0.5):
    total = n_a + n_b
    return stats.chisquare([n_a, n_b], [total * expected_ratio, total * (1 - expected_ratio)]).pvalue
```

**Novelty and primacy effects:** users react to change itself at first (clicking a new widget out of curiosity, or avoiding an unfamiliar layout). Plot the effect by day since exposure; run long enough to see it stabilise.

**Other threats:** interference between units (2.9), seasonality and holidays, other experiments running at once, survivorship (analysing only users who reached a page that the treatment changed).

## 2.5 Variance reduction with CUPED

**CUPED** (Deng et al., Microsoft, 2013) uses pre-experiment data to remove predictable variance. For each user, adjust the metric with their pre-period value of a correlated covariate X (often the same metric before the test):

`Y_cuped = Y − θ(X − mean(X)), θ = cov(Y, X) / var(X)`

Randomisation makes X independent of treatment, so the effect estimate stays unbiased while its variance shrinks by a factor of (1 − ρ²), where ρ is the correlation between X and Y.

```python
theta = np.cov(post, pre)[0, 1] / np.var(pre, ddof=1)
post_cuped = post - theta * (pre - pre.mean())
# then run the usual test on post_cuped instead of post
```

In a simulation with a correlation of about 0.82 between pre- and in-test order value, the standard error of the effect fell from 0.197 to 0.113, a 67% variance reduction, exactly 1 − ρ². That is equivalent to running the test with about three times the users. New users have no pre-period: use 0 with an indicator, or other covariates. The regression version (ANCOVA, Y ~ treatment + X) is equivalent and extends to several covariates.

## 2.6 Multiple testing

Test 20 metrics at α = 0.05 with no real effect and you expect one false positive. Segments, metrics, variants and repeated looks all multiply the chances.

- **Bonferroni:** test each of m hypotheses at α/m. Controls the chance of any false positive (family-wise error); very conservative.
- **Benjamini-Hochberg:** controls the **false discovery rate** (the expected share of false positives among discoveries); more power, the usual choice for many metrics or segments.

```python
def benjamini_hochberg(p_values, q=0.05):
    p = np.asarray(p_values); m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m          # compare k-th smallest with q·k/m
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    reject = np.zeros(m, bool); reject[order[:k]] = True
    return reject

# p-values 0.001, 0.008, 0.039, 0.041, ...: BH rejects the first two; Bonferroni (0.005) only the first
```

The practical defence: one pre-registered primary metric, a small set of guardrails, and segment analyses labelled as exploratory.

## 2.7 Sequential testing and peeking

Checking results every day and stopping the first time p < 0.05 inflates false positives dramatically. In a simulated A/A test (no real difference) checked daily for 20 days, about 25% of runs reached "significance" at some point, five times the nominal 5%.

Options:

- **Fixed horizon:** compute the sample size, run to the end, look once. Simple; monitor only guardrails meanwhile.
- **Group sequential designs:** a few planned interim looks with stricter thresholds (O'Brien-Fleming boundaries) that preserve overall α.
- **Always-valid inference:** mixture sequential probability ratio tests (mSPRT) and confidence sequences let you look at any time; used by several experimentation platforms.
- **Bayesian analysis** with a decision rule on expected loss; still needs discipline about stopping.

## 2.8 Interleaving for ranking

For comparing two rankers (search, recommendations), interleaving merges both rankings into one list per user and credits clicks or orders to the ranker that contributed each item. Because each user sees both systems, between-user variance disappears, and interleaving typically needs far less traffic than an A/B test to find the better ranker. Use it to screen candidates; confirm business impact with an A/B test (Chapter 44.9).

## 2.9 Experiments in two-sided marketplaces

In a delivery marketplace, users in treatment and control **share couriers and restaurants**. If a new dispatch algorithm grabs the nearest couriers for treatment orders, control orders get worse couriers: the measured difference overstates the real effect (and the effect at 100% rollout may even be negative). This violates the assumption that one unit's treatment does not affect another (SUTVA).

**Designs that contain the interference:**

![Diagram](images/02-01.png)

*switchback design · 3 cities × 8 time slots, illustrative assignment*

- **Switchback tests:** randomise **time slots within a region** (for example 30–60 minute blocks per city). Everything in the slot gets the same treatment, so arms never compete for supply. Watch for carryover between slots (orders assigned in one slot delivered in the next): use buffers or discard boundary periods, and analyse at the slot level with clustered standard errors.
- **Geo (cluster) experiments:** randomise whole cities or zones. Clean but few units, so low power; use matched pairs or synthetic controls.
- **Two-sided randomisation** or budget-split designs for pricing and ads.

User-level A/B tests remain fine for changes whose effect stays on the demand side and does not draw on shared supply (most UI changes, ranking within what is available, notification copy). The senior skill is recognising which kind of change you have. Chapter 45 covers marketplace experiments further.

## 2.10 Causal inference basics

When you cannot randomise, you estimate effects from observational data, with assumptions.

**Difference-in-differences (DiD):** compare the change over time in a treated group with the change in a control group. Assumes **parallel trends**: without treatment, both would have moved alike. Check pre-period trends.

```python
import statsmodels.formula.api as smf
# df: one row per city-period with y, treated (0/1) and post (0/1)
model = smf.ols("y ~ treated * post", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["city"]})
model.params["treated:post"]          # the DiD estimate
```

**Inverse propensity weighting (IPW):** model the probability of treatment given confounders (the propensity score), then weight each unit by the inverse of the probability of the treatment it actually got, so the weighted groups look alike on observed confounders.

```python
from sklearn.linear_model import LogisticRegression

ps = LogisticRegression().fit(X_confounders, t).predict_proba(X_confounders)[:, 1]
ps = np.clip(ps, 0.01, 0.99)                                   # avoid extreme weights
ate = np.mean(t * y / ps) - np.mean((1 - t) * y / (1 - ps))
```

In a simulation where heavy users are more likely to get a promotion and the true effect is 1.0, the naive difference in means was 3.1 and the IPW estimate 1.03. IPW only fixes **observed** confounders; unobserved ones still bias it.

**Uplift modelling** estimates the effect per user, to target treatments (vouchers, push notifications) at people they actually change: T-learner (separate outcome models for treated and control, difference of predictions), X-learner, causal forests. Train on randomised data where possible, and evaluate with uplift (Qini) curves.

Other tools to name: regression discontinuity, instrumental variables, synthetic control, and causal graphs (DAGs) to reason about which variables to adjust for.

## 2.11 Statistics for model evaluation

The same tools apply when the "experiment" is a comparison between models rather than between product variants, with three differences covered in depth in Chapter 23:

- **The unit is a question, not a user**, and questions are often clustered (several per passage, repository or template), so standard errors must be clustered.
- **Comparisons are paired**: two models answer the same questions, so analyse per-question differences (McNemar for binary outcomes, a paired bootstrap otherwise). Pairing removes shared difficulty and makes small differences detectable.
- **Training is random**: the same recipe trained with a different seed gives a different model. Seed variance is a second source of noise on top of question sampling, and for small models it is often the larger one (Chapter 34).

Rule of thumb: a difference between two models smaller than twice its paired standard error, or smaller than the seed-to-seed spread of the baseline, is not a result.

## Pitfalls

- **Peeking and stopping early** when the primary metric looks significant.
- **No power calculation,** then calling a non-significant result "no effect".
- **Randomising by user but analysing by order** without clustered errors.
- **Ignoring sample ratio mismatch.**
- **Testing many metrics and segments** and reporting the significant ones.
- **User-level A/B tests for supply-side changes** (dispatch, pricing, courier incentives).
- **Short tests** that capture novelty effects or miss weekly cycles.
- **Confusing statistical and practical significance.**
- **Observational comparisons without a causal design** ("users who used the feature order more").

## Interview questions

Questions 1–5 basic, 6–11 intermediate, 12–15 senior.

**1. What is a p-value?** The probability of observing a result at least as extreme as this one if there were no real effect. Not the probability that the null is true, and not the effect size.

**2. What is statistical power and why does it matter?** The probability of detecting an effect of a given size if it exists. Underpowered tests miss real effects and, when they do find something, exaggerate its size.

**3. How do you compute the sample size for an A/B test?** From the baseline rate or variance, the minimum detectable effect, α and power: n ≈ (zα/2 + zβ)² · variance terms / MDE² per arm. Halving the MDE quadruples n.

**4. What are guardrail metrics?** Metrics you commit not to harm (latency, cancellations, delivery time, revenue); a launch needs the primary metric up and no significant guardrail degradation.

**5. How do you build a confidence interval for a median?** Bootstrap: resample the independent units with replacement, recompute the median, take the 2.5th and 97.5th percentiles.

**6. What is CUPED and why does it work?** Adjust each user's outcome by a pre-experiment covariate times θ = cov(Y, X)/var(X); randomisation keeps the estimate unbiased, and variance falls by 1 − ρ².

**7. Your test was significant on day 3 of a planned 14. Do you stop?** No: early peeking inflates false positives, and early effects often include novelty. Continue to the planned horizon, or use a sequential method designed for interim looks.

**8. What is sample ratio mismatch?** The observed split differs significantly from the planned one, signalling broken assignment or logging; results are untrustworthy until it is explained.

**9. How do you handle 15 secondary metrics?** Pre-register one primary metric; treat secondaries as supporting evidence; control the false discovery rate with Benjamini-Hochberg; label segment findings exploratory and confirm in follow-up tests.

**10. Why is interleaving more sensitive than A/B testing for rankers?** Each user sees both rankers in one list, so preference is measured within users, removing between-user variance.

**11. What is difference-in-differences and its key assumption?** The change in the treated group minus the change in the control group; it assumes parallel trends without treatment, checked on pre-period data.

**12. How would you test a new courier dispatch algorithm?** User-level randomisation is invalid because arms share couriers. Use a switchback design (city × time slots) or geo randomisation, with slot-level analysis, clustered errors, buffers for carryover, and metrics such as delivery time, courier utilisation, cancellations and orders per hour.

**13. The A/B test shows +3% orders but the effect disappears after full rollout. Why?** Novelty; interference (treatment took supply from control, inflating the difference); a winner's-curse overestimate from an underpowered test; seasonality; or a changed mix of users at full rollout.

**14. How would you measure the effect of a loyalty programme that was launched without an experiment?** Difference-in-differences against comparable cities or users not yet eligible, matched or propensity-weighted comparisons on observed confounders, checking pre-trends, and sensitivity analyses for unobserved confounding; ideally, randomise the next expansion.

**15. How do you target vouchers to maximise incremental orders?** Uplift modelling on randomised voucher data (T- or X-learner, causal forests), ranking users by predicted incremental effect rather than predicted ordering probability; evaluate with Qini curves and a holdout test.

## Practice

1. **Implement** the two-proportion z-test, Welch's t-test, sample size functions, SRM check, CUPED and Benjamini-Hochberg from scratch; check against `scipy` and `statsmodels`.
2. **Peeking simulation:** reproduce the A/A peeking experiment and plot false positive rate against number of looks.
3. **CUPED:** on a public e-commerce dataset with user histories, simulate an experiment and measure variance reduction from pre-period spending.
4. **Switchback simulation:** simulate a small city where treatment orders take the nearest couriers, compare user-level and switchback estimates against the true global effect.
5. **Causal:** reproduce the IPW example; then add an unobserved confounder and watch IPW fail.
6. **Interview drill:** design the experiment for E.1 (restaurant feed) and E.8 (demand forecasting changes) in five minutes each.

## Read next

- Kohavi, Tang and Xu, *Trustworthy Online Controlled Experiments* (2020): chapters on metrics, sample ratio mismatch, threats to validity, variance reduction.
- Deng et al., *Improving the Sensitivity of Online Controlled Experiments by Utilizing* *Pre-Experiment Data* (CUPED, 2013).
- Johari et al., *Peeking at A/B Tests: Why It Matters, and What to Do About It* (2017).
- DoorDash and Lyft engineering blogs on switchback experiments; Bojinov, Simchi-Levi and Zhao, *Design and Analysis of Switchback Experiments* (2020).
- Angrist and Pischke, *Mostly Harmless Econometrics*, or Cunningham, *Causal* *Inference: The Mixtape* (free online), for DiD, IV and RD.
- Gutierrez and Gérardy, *Causal Inference and Uplift Modeling: A Review of the* *Literature* (2017).

**Links:** [Trustworthy Online Controlled Experiments](https://experimentguide.com/) · [Evan Miller's A/B testing tools and essays](https://www.evanmiller.org/ab-testing/sample-size.html) · [Seeing Theory](https://seeing-theory.brown.edu/). Other titles above can be found by searching their title.

## Done when

You can explain p-values, intervals and power precisely, including what they are not. You can compute sample sizes for proportions and means and explain the MDE trade-off. You can design an A/B test end to end, including SRM checks and guardrails. You can implement CUPED and explain its variance reduction.

You can handle multiple testing and explain why peeking is a problem. You can design a switchback or geo experiment for a marketplace change. You can explain and code DiD and IPW, and when uplift modelling is needed.

---

[← 1 · The maths you actually use](01-the-maths-you-actually-use.md) · [Contents](../README.md#contents) · [3 · ML fundamentals →](03-ml-fundamentals.md)
