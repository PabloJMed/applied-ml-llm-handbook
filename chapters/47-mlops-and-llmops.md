# 47 · MLOps and LLMOps

Part XI · Applied ML systems (optional track). Prerequisites: Chapter 42 (infrastructure primer). Written for readers who have trained models but never run one in production.

## Why it matters

Most of a model's life happens after the notebook. Data changes, the model has to be retrained, new versions must reach users without breaking anything, and someone must notice when predictions quietly go wrong. **MLOps** is the set of practices and tools that make this repeatable: versioned data and code, automated pipelines, safe releases, monitoring and quick rollback. **LLMOps** adds what is special about LLM features: prompts as code, evaluation sets in CI, traces of every call, and provider updates you did not schedule. Senior data scientists are expected to own a model end to end, or design it so a platform team can run it. Interviews probe failure modes ("accuracy dropped this week; what do you check?") and release discipline ("how do you ship a new prompt?").

The code below was run; outputs are quoted.

## Key concepts

- **ML lifecycle**: data → features → training → evaluation → registration → deployment → monitoring → back to data. Google's [MLOps levels guide](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) describes how teams automate it step by step.
- **Technical debt in ML**: models create hidden maintenance costs through data dependencies and feedback loops. The classic paper: [Hidden Technical Debt in ML Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/).
- **Feature store**: one definition of each feature, served offline for training and online for prediction. [Feast documentation](https://docs.feast.dev/).
- **Training–serving skew**: a feature computed differently in training and in production.
- **Experiment tracking and model registry**: a record of every training run and every model version with its stage. [MLflow documentation](https://mlflow.org/docs/latest/index.html).
- **Shadow, canary and A/B releases**: ways to expose a new version gradually. [Canary release](https://martinfowler.com/bliki/CanaryRelease.html) (Martin Fowler).
- **Data drift and concept drift**: inputs change, or the relation between inputs and outcome changes. [Concept drift](https://en.wikipedia.org/wiki/Concept_drift).
- **Monitoring tools**: open-source libraries such as [Evidently](https://docs.evidentlyai.com/) compute drift and quality reports.
- **Behavioural tests**: checks of model behaviour such as invariance ("changing a name must not change the label"). [CheckList paper](https://arxiv.org/abs/2005.04118).
- **LLM tracing**: logging each LLM call, retrieval and tool call as a linked trace, with tokens, cost and prompt version.

![Diagram](images/29-01.png)

*Every arrow is a place things break silently; MLOps is the practice of making each one visible and repeatable.*

## 47.1 Where MLOps starts: from notebook to pipeline

In a notebook you load data, train, look at a metric and save a file. That works once. It fails in production because nobody can rerun it the same way, it depends on your laptop's libraries, and nothing checks the data. The first step of MLOps is turning the notebook into a **pipeline**: a sequence of steps, each a function with clear inputs and outputs, run by a scheduler.

1. **Extract** the training data from the warehouse for a date range.
2. **Validate** it: schema, null rates, value ranges, row counts versus yesterday (Chapter 48.6). Stop here if it fails.
3. **Build features** with the same code used in production (47.2).
4. **Train** the model with a config file holding all hyper-parameters.
5. **Evaluate** on a fixed held-out set and on slices (city, language, new vs returning users).
6. **Register** the model with its metrics and lineage if it passes the gates (47.3).

An **orchestrator** (Airflow, Dagster, Prefect, Kubeflow, Metaflow, or the cloud's own) runs the steps in order, retries failures and alerts on errors. Each step runs in a container (Chapter 42), so it behaves the same everywhere.

For every run, record: the code version (git commit), the data snapshot or date range, the config, the container image, the metrics and the produced files. **Experiment tracking** tools (MLflow, Weights & Biases) do this with a few lines of logging. The test of reproducibility: can a colleague rebuild last month's production model from the registry entry alone?

## 47.2 Feature stores and training–serving skew

A model is only as good as the features it sees *at prediction time*. A common, silent bug: the training data computes "restaurant cancellation rate over 7 days" in SQL, while the production service computes it in Python with a slightly different window or a different treatment of missing values. Offline metrics look great; online results disappoint. This is **training–serving skew**.

A **feature store** fixes it by keeping one definition per feature and materialising it twice:

- an **offline store** (warehouse tables) for training, with **point-in-time** lookups so each training row only sees values that existed at that moment (Chapter 48.3);
- an **online store** (a fast key-value store such as Redis) that the prediction service reads in a millisecond.

Other benefits: features are reused across models (ETA, ranking and fraud can all use the same restaurant features), and there is a catalogue of what exists. Tools: Feast (open source), Tecton, and the feature stores in Databricks, SageMaker and Vertex AI. Without a feature store, the cheapest defence is to **log the exact features used at prediction time** with every prediction; that log becomes your best training data and your first debugging tool.

## 47.3 Model registry, CI/CD and quality gates

A **model registry** is a catalogue of model versions: the file location, the metrics, the training run that made it, and a **stage** (candidate, staging, production, archived). Promotion is a decision backed by evidence: the evaluation report, the comparison with the current production model, and a sign-off.

CI/CD (Chapter 42.10) for ML has three loops:

- **Code CI** on every pull request: unit tests for feature functions and pre-processing, linting, type checks.
- **Model CI**: train on a small sample to catch breakage; evaluate the candidate on fixed held-out sets with **gates**: no worse than production by more than x on the main metric, no slice worse by more than y, latency within budget, model size within memory limits.
- **CD**: build the serving container with the registered model, deploy behind a feature flag, then follow the release ladder (47.4).

**Behavioural tests** add checks that metrics miss: *invariance* (changing a restaurant's name must not change its cuisine label), *directional expectations* (a longer distance must not lower the ETA), and *minimum functionality* (obvious cases must be right).

## 47.4 Releasing safely: the release ladder

Never switch 100% of traffic to a new model at once. Climb a ladder, stopping if anything looks wrong:

| Step | How | What it catches | Cost |
|---|---|---|---|
| 1. Shadow | The new model receives a copy of live traffic; its outputs are logged, not shown | Crashes, latency, strange output distributions, disagreement with production | Extra compute; no user impact |
| 2. Canary | 1–5% of real traffic uses it, with automatic rollback rules | Errors and guardrail breaks on real users, early | Small user risk |
| 3. A/B test | Randomised split, long enough to measure business metrics | Causal impact on orders, conversion, retention (Chapter 2) | Time and traffic |
| 4. Full rollout | 100%, old version kept ready for instant rollback | — | — |

For rankers, **interleaving** (mixing results of both rankers in one list and counting which get clicked) detects preference with far less traffic than an A/B test. **Bandits** shift traffic towards the better variant while still learning, useful with many variants.

A canary's automatic rollback rule is a pre-agreed statistical test. Example: is the canary's error rate worse than production's?

```python
import numpy as np
from scipy import stats

def two_prop(x1, n1, x2, n2):              # control errors/requests, canary errors/requests
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (x2 / n2 - x1 / n1) / se
    return z, 1 - stats.norm.cdf(z)         # one-sided: is the canary worse?

two_prop(120, 95_000, 21, 5_000)
# control 0.13% errors, canary 0.42% -> z = 5.39, p = 3e-8 -> roll back automatically
```

**Rollback** must take minutes: keep the previous container and model version deployed or one command away, and switch with a flag rather than a rebuild.

## 47.5 Monitoring: four layers

A model can be "up" (the server answers) and badly wrong. Monitor four layers, from the fastest signal to the slowest:

1. **System health**: latency p50/p95/p99, error rate, throughput, CPU/GPU memory, cost. Standard service monitoring (Chapter 42.11).
2. **Data quality**: input schema, null rates, ranges, and the distribution of each feature compared with training.
3. **Model behaviour**: the distribution of predictions and confidence; once labels arrive, accuracy and calibration by slice.
4. **Business outcome**: the metric the model exists for: orders, on-time delivery rate, contact rate.

**Kinds of drift**, with delivery examples:

- **Covariate shift**: inputs change. Ramadan changes when and what people order; a new city launch brings new restaurants.
- **Prior (label) shift**: the base rate changes. Fraud attempts spike during a big promotion.
- **Concept drift**: the relation between inputs and outcome changes. A new delivery mode (bikes instead of cars) makes old ETA patterns wrong.

Labels often arrive late (an ETA label in an hour, a churn label in a month), so input and prediction drift are the early warnings. The **population stability index (PSI)** is the standard tabular drift score: bin the training distribution, compare the share of live data in each bin.

```python
def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))   # bins from training data
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

# basket value, training distribution N(30, 8):
# same distribution         -> 0.0007
# mean +2                   -> 0.059
# mean +6 and wider spread  -> 0.453
```

Rule of thumb: below 0.1 stable, 0.1–0.25 watch, above 0.25 investigate. Avoid alerting on p-values of statistical tests: with large samples, a Kolmogorov–Smirnov test calls even the harmless +2 shift "significant" (p = 1e-27 on 5,000 rows each). Alert on effect size, and compare with the same season last year before raising alarms about Ramadan or summer.

**When to retrain**: on a schedule (simple and predictable), on a trigger (drift or performance past a threshold), or continuously for fast-moving targets. Always pass the new model through the same gates and release ladder. Fix data bugs before retraining; retraining on broken data just learns the bug.

## 47.6 LLMOps: what changes with LLM features

LLM features follow the same lifecycle, with new artefacts and new failure modes.

- **More things to version.** The prompt (system prompt, templates, few-shot examples), the model name and version, decoding settings, the retrieval configuration (embedding model, chunking, top-k), and tool definitions. A change to any of them is a deployment and goes through the release ladder. Store prompts as files in git, not as strings scattered in code.
- **Provider updates are deployments you did not schedule.** Pin model versions where the API allows it; re-run your evaluation set when a provider announces changes; keep a fallback model.
- **Evaluation in CI.** A golden set of inputs with expected properties, scored automatically and with LLM-as-judge (Chapter 24), with thresholds that block a merge. Include safety and prompt-injection tests (Chapter 28).
- **Tracing.** Log every call as a span in a trace: inputs, retrieved documents, tool calls, outputs, tokens, latency, cost, prompt version, with personal data redacted. Tools: LangSmith, Langfuse, Arize Phoenix, MLflow tracing, or OpenTelemetry-based setups.
- **Online signals.** Thumbs up/down, user edits of generated text, escalations to a human agent, retries, conversation length, and sampled LLM-as-judge scoring of live traffic.
- **Cost monitoring.** Tokens per request by feature and prompt version, with alerts on spikes: an agent stuck in a loop can burn a monthly budget overnight. Chapter 21 lists the levers.

## 47.7 Ownership and on-call

For each model in production, write down: owner, purpose, inputs and their sources, metrics and thresholds, dashboards, alert routes, retraining schedule, rollback procedure, and known limitations. This **model card plus runbook** fits on one page and is worth more than a perfect platform nobody reads. After each incident, run a short blameless post-mortem and add a test or monitor that would have caught it.

## 47.8 A maturity path

Teams rarely build everything at once. A practical order:

1. **Reproducible training**: code in git, a pipeline instead of a notebook, experiment tracking.
2. **Prediction logging** with model version and features, and basic system dashboards.
3. **A registry and evaluation gates** before deploy.
4. **Shadow and canary releases** with automatic rollback.
5. **Drift and business-metric monitoring** with alerts and a runbook.
6. **A feature store** once several models share features or skew bugs keep appearing.
7. **Automatic retraining** once the steps above are trusted.

Google's MLOps guide calls these levels 0 to 2; whatever the naming, saying where a team is and what you would add next is a strong interview answer.

## 47.9 Operations for research-scale training

Research teams run a different kind of production: long training jobs and large experiment fleets.

- **Run tracking**: every run's configuration, code commit, data version and metrics in one place; runs grouped by experiment.
- **Training-health monitoring**: loss, gradient norms, logit statistics, throughput and MFU, with alerts on spikes, stalls and NaNs (Chapters 7 and 17).
- **Checkpoint management**: sharded, asynchronous, retained on a schedule; automatic evaluation of each checkpoint on a fixed suite.
- **Fault tolerance**: automatic restarts, node health checks, straggler detection, exact data-loader resumption.
- **Reproducibility**: pinned environments (containers), fixed data snapshots, recorded seeds; documented non-determinism.
- **Evaluation pipelines** that score checkpoints automatically and publish dashboards, so model comparisons use the same code every time (Chapter 23).
- **Cost accounting**: GPU-hours per project and per experiment, to support the compute budgeting of Chapter 34.

## Pitfalls

- Different feature code for training and serving.
- Monitoring only system metrics: the model can be up and wrong.
- Alerting on p-values of drift tests with large samples.
- No seasonal baseline: comparing this week with last week across a holiday.
- Unpinned LLM versions and unversioned prompts.
- A release process that cannot roll back in minutes.
- Retraining automatically on data that failed validation.

## Interview questions

**Mid level**

1. What is training–serving skew and how does a feature store help?
2. Explain shadow, canary and A/B deployments.
3. What is PSI and how do you read it?
4. What would you log with every prediction?

**Senior level**

5. Your ETA model's error jumped 20% this week. Walk through your investigation.
6. Design the CI/CD process for a prompt-based classifier used by three teams.
7. Labels arrive 30 days late. How do you monitor the model in the meantime?
8. How would you decide when to retrain?

Short answers: (5) Check system health and recent deployments first; then input pipelines (nulls, schema, a broken upstream feature); then drift by feature and by city; then external events (weather, a promotion, a new delivery mode); then compare with the same season last year. Fix data bugs before retraining. (6) Prompts in git; an evaluation set with gates in CI; a pinned model version; shadow then canary with guardrails; traces tagged by prompt version; owners notified on change; rollback by flag. (7) Input and prediction drift, proxy labels (early signals such as complaints), a small fast-labelled sample, and business metrics. (8) A schedule plus triggers; retrain when drift or performance crosses a threshold and the new model passes the gates.

## Practice

- Turn one of your notebooks into a pipeline of functions with a config file, and log runs to MLflow.
- Add PSI monitoring to three features of a model and backfill it over a year; mark holidays and promotions.
- Write a one-page model card and runbook for a model you know.
- Build a tiny GitHub Actions job that runs a 100-example evaluation of a prompt and fails below a threshold.
- Run a shadow comparison of two prompts over a day of logged inputs and analyse the disagreements.

## Go deeper

- [MLOps: continuous delivery and automation pipelines in machine learning](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (Google Cloud): the maturity levels.
- Sculley et al., [Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/) (2015).
- Chip Huyen, [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) (O'Reilly, 2022): deployment, monitoring and continual learning chapters.
- [Made With ML](https://madewithml.com/): a free end-to-end MLOps course.
- Ribeiro et al., [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118) (2020).
- Tool docs: [MLflow](https://mlflow.org/docs/latest/index.html), [Feast](https://docs.feast.dev/), [Evidently](https://docs.evidentlyai.com/).
- [Canary release](https://martinfowler.com/bliki/CanaryRelease.html) (Martin Fowler).

## Done when

You can draw the lifecycle loop and name what can break at each step. You can explain training–serving skew, feature stores and point-in-time correctness to a beginner.

You can describe the release ladder and write a canary rollback test. You can compute PSI and explain why p-values mislead at scale. You can explain how prompt and model changes are versioned, tested and rolled out. You can place a team on the maturity path and say what to add next.

---

[← 28 · ML system design](28-ml-system-design.md) · [Contents](../README.md#contents) · [30 · Data for ML →](30-data-for-ml.md)
