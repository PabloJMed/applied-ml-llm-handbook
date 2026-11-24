# 25 · RAG and agent evaluation

Part VII · Evaluation science. Applies Chapter 24's methods to Chapter 31 (RAG) and Chapter 33 (agents). Worked designs that depend on it: E.4 RAG support assistant and E.10 agentic customer-service flow.

## Why it matters

RAG systems and agents have several stages, and an end-to-end score ("72% of answers are correct") tells you that something is wrong, not what. Senior engineers evaluate each stage separately so they know where to spend effort: retrieval, context assembly, generation, tool use. Interviewers often ask "how would you evaluate this?" right after a RAG or agent design, and a staged answer with concrete metrics is what they look for.

Agents add a second difficulty: they act. An agent's success is a fact about the world it changed, so its evaluation needs a sandboxed world, a checker of the final state, repeated runs because the agent is stochastic, and a grader the agent cannot fool. That last point has become central since agents started being trained with reinforcement learning against the same kind of checkers: a grader with a loophole produces an agent that finds it. At frontier labs, "how would you know the agent did not cheat?" is now a standard follow-up.

The chapter starts with the choice between evaluating each component and evaluating the whole system (25.1); a good suite does both. Retrieval is measured with recall and precision of the right context (25.2), generation with faithfulness, citations and answer quality (25.3), and 25.4 explains what the popular frameworks compute and where their numbers mislead. Agents need different measures, because they take several steps and change the world (25.5), including coding and computer-use agents, simulated users and safety under attack. Section 25.6 puts the pieces into one evaluation suite, and 25.7 works through an agent evaluation with real numbers, including cost. Section 25.8 covers how to harden agent graders against exploits.

The reason for component metrics is diagnosis. An end-to-end score says whether the system works; only per-component scores say which part to fix when it does not. A wrong answer from a RAG system can come from retrieval that missed the right passage, or from generation that ignored it, and the fixes are completely different.

## Key concepts

- **Component vs end-to-end evaluation**: measure retrieval and generation separately to find where a RAG system fails, then measure the whole answer.
- **Retrieval metrics**: recall@k (did the right passage come back?), MRR and nDCG (how high?).
- **Generation metrics for RAG**: faithfulness (claims supported by the context), answer relevance, context precision and recall, citation precision and recall. [Ragas paper](https://arxiv.org/abs/2309.15217), [Ragas documentation](https://docs.ragas.io/), [ALCE](https://arxiv.org/abs/2305.14627).
- **Synthetic test sets**: generating questions from your documents with an LLM to bootstrap evaluation, then checking a sample by hand.
- **Agent evaluation**: task success rate, number of steps, cost, tool-call correctness and the trajectory, ideally in a simulated environment with a user model. [τ-bench](https://arxiv.org/abs/2406.12045).
- **pass@k and pass^k**: at least one of k runs succeeds, versus all k succeed; reliability for users is pass^k.
- **Grader hardening**: making end-state checks and test suites robust to agents that edit tests, read leaked answers or cause unchecked side effects.
- **Cost-aware evaluation**: reporting accuracy against cost and comparing with simple baselines. [AI Agents That Matter](https://arxiv.org/abs/2407.01502).
- **Error analysis**: reading failures and grouping them into categories before building metrics. [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/).

## 25.1 Component vs end-to-end evaluation

A RAG system or an agent is a chain of steps, and a failure at any step produces the same symptom: a bad final answer. End-to-end metrics tell you how often that happens. Component metrics tell you where, which is what you need in order to fix it.

![RAG evaluation · one metric per link](images/25-diagram-1.png)

*RAG evaluation · one metric per link*

| Link | Question it answers | Needs |
|---|---|---|
| Question → context | Did retrieval find the passages that answer the question, and rank them high? | Relevance labels per question, or an LLM judge |
| Context → answer | Is every claim in the answer supported by the context (faithfulness, also called groundedness)? | Only the context and the answer: no reference needed |
| Question → answer | Does the answer address what was asked (answer relevance)? | Only the question and answer |
| Answer → reference | Is the answer correct and complete? | A reference answer or required facts |

**End-to-end** metrics (correctness, task success, user satisfaction) tell you whether the system works. **Component** metrics tell you why it does not. You need both: end-to-end to decide whether to ship, components to decide what to fix. A useful habit is to tag every failed case with the first stage that went wrong (Chapter 31.9).

## 25.2 Retrieval metrics: context recall and precision

The generator can only use what retrieval gives it. If the passage containing the answer is not retrieved, no prompt or model will produce a grounded answer; if it is buried among irrelevant passages, the model may miss it. Retrieval metrics measure both risks separately.

With relevance labels per question (which passages or documents answer it), use the standard retrieval metrics from Chapter 30.9. In brief: **recall@k** is the share of relevant passages found in the top k. **MRR** (mean reciprocal rank) averages 1/rank of the first relevant passage over questions, so a first hit at rank 3 scores 0.33. **nDCG@k** sums the relevance of each result discounted by its position, DCG@k = Σᵢ relᵢ / log₂(i + 1), and divides by the best achievable DCG for that question, so 1.0 means the ideal order. A single relevant passage at rank 3 gives nDCG = (1/log₂ 4)/(1/log₂ 2) = 0.5. Two names you will meet in RAG tooling:

- **Context recall:** the share of the needed information that appears in the retrieved context. Label-based version: relevant passages retrieved ÷ relevant passages. Reference-based version (as in RAGAS): split the reference answer into claims and check how many are supported by the context.
- **Context precision:** how much of the context is relevant, rewarding relevant passages ranked early (an average-precision style score).

```python
def context_recall(retrieved_ids, relevant_ids):
    return len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else None

def context_precision(retrieved_ids, relevant_ids):
    """Average precision over the ranked context."""
    hits, total = 0, 0.0
    for i, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant_ids:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0
```

A check by hand: for the ranked context [a, b, c, d] with b and d relevant, the precision at the two hits is 1/2 and 2/4, so context precision is (0.5 + 0.5)/2 = 0.5. Moving b to the top would give (1 + 2/4)/2 = 0.75.

Measure these **at the context the LLM actually receives** (after re-ranking and assembly), as well as at the retriever's output, so you can tell a retriever miss from an assembly drop. Without labels, an LLM judge can rate each retrieved passage as relevant or not to the question, which gives precision but not recall (it cannot know what was missed).

## 25.3 Generation metrics

Given the retrieved context, the generator can still fail in two independent ways: it can say things the context does not support, or it can fail to answer the question that was asked. The metrics below measure each failure separately, because they have different causes and different fixes.

**Faithfulness (groundedness).** Is every claim in the answer supported by the retrieved context? The standard method decomposes the answer into atomic claims, then checks each against the context with an LLM judge or an entailment (NLI) model. The score is supported claims ÷ total claims, and the unsupported claims themselves are the useful output for debugging.

```python
def faithfulness(answer, context, extract_claims, is_supported):
    """extract_claims and is_supported are LLM-judge (or NLI) calls in practice."""
    claims = extract_claims(answer)
    if not claims:
        return None, []
    verdicts = [(c, is_supported(c, context)) for c in claims]
    return sum(v for _, v in verdicts) / len(claims), [c for c, v in verdicts if not v]

# "Vouchers are worth 20% of the order. Refunds arrive within 3 days."
# against a context that never mentions 3 days -> (0.5, ['Refunds arrive within 3 days'])
```

Faithfulness is not correctness: an answer can faithfully repeat an outdated policy that was retrieved. And it is not a hallucination detector for claims the model got right from its own knowledge but which are not in the context; decide whether those count as failures for your product (for policy answers, they usually should).

**Limits of NLI checkers.** Off-the-shelf NLI models were trained on single-sentence premise and hypothesis pairs. They degrade when the premise is a long context, because the relevant sentence is diluted, and when a claim needs several sentences combined ("the voucher is 20% of the order" plus "the order was €40" supports "the voucher is €8", but neither sentence does alone). They also struggle with numbers, negation and tables. Use them on the shortest span that could support the claim (retrieve the best sentences first), or use an LLM judge for multi-hop claims, and calibrate either against human labels (Chapter 24.7).

**Answer relevance.** Does the answer address the question? Judge it directly, or use the RAGAS method: an LLM generates several questions that the answer would be a good answer to, each is embedded, and the score is the mean cosine similarity between those generated questions and the original question. The idea is that an on-topic answer implies the original question, while an evasive or off-topic answer implies different questions. It catches evasive and off-topic answers, but it fails on answers that are correct and terse. "Yes" or "€8" implies almost any question, so the generated questions are vague and the similarity is low even though the answer is perfect. It also rewards answers that restate the question. Treat low scores on short answers as a prompt to read, not as a failure.

**Correctness.** Against a reference answer or a list of required facts: an LLM judge checks that the required facts are present and no contradicting facts are stated. For short factual answers, exact match or token F1 (Chapter 24.4).

**Citations.** When the system cites sources, evaluate the citations separately from the answer: a correct answer with a wrong citation teaches users to trust the wrong document. ALCE (Gao et al., 2023) defines two metrics per generated statement. **Citation recall** asks whether the cited passages, taken together, entail the statement: if not, the statement is unsupported by its citations. **Citation precision** asks, for each citation, whether it is needed: a citation is irrelevant if it does not support the statement on its own and removing it does not hurt the support given by the others. Both are measured with an NLI model or a judge. A system that cites every retrieved passage scores high on citation recall and low on precision, which is the failure to watch for. Add a code check that every cited id exists in the context (Chapter 24.5).

**Abstention.** A RAG system must answer when the documents contain the answer and abstain when they do not. Include out-of-scope questions in the evaluation set and report:

```python
import numpy as np

def abstention_report(should_answer, did_answer, correct):
    sa, da, c = (np.asarray(x, dtype=bool) for x in (should_answer, did_answer, correct))
    return {
        "correct_when_answered": c[sa & da].mean() if (sa & da).any() else None,
        "wrongly_abstained": (~da[sa]).mean() if sa.any() else None,    # missed answers
        "wrongly_answered": da[~sa].mean() if (~sa).any() else None,    # answered out of scope
    }
```

The last number, answering when it should not, is usually the most dangerous for customer-facing systems, and generic RAG metrics do not show it.

## 25.4 RAGAS-style frameworks and their pitfalls

Frameworks such as RAGAS, DeepEval and TruLens package these metrics (faithfulness, answer relevance, context precision and recall) as ready-made LLM-judge prompts. They are a fast start. The pitfalls:

- **Uncalibrated judges.** The scores come from LLM prompts you did not write and have not checked against human labels on your data. Calibrate them like any judge (Chapter 24.7) before trusting a number.
- **Generic criteria.** They measure generic properties, not your failure modes ("cited the wrong country's policy", "gave a refund amount above the cap"). Add your own checks.
- **Score drift with the judge model.** Changing the underlying LLM changes the scores. Pin versions.
- **Synthetic test sets.** Generated questions are cleaner than real ones and flatter the system (Chapter 30.9).
- **Averages that hide abstention failures.** A system that answers everything can score well on faithfulness and relevance while answering out-of-scope questions.

Use them as scaffolding: borrow the metric definitions, swap in calibrated judges, add product-specific checks, and always read the failures.

## 25.5 Evaluating agents

Agents take several steps, call tools and change the world (Chapter 33). Evaluate at three levels:

**1. Outcome (task success).** Did the agent achieve the goal? Best checked on the **final state** of a sandboxed environment rather than on the agent's words: the voucher issued equals the policy amount, the ticket is closed, nothing else changed. This is the approach of agent benchmarks such as τ-bench (simulated customer-service tasks with a database and policies).

```python
def check_end_state(state, expected):
    """Compare the final environment state with the expected outcome; ignore unspecified fields."""
    return {k: state.get(k) == v for k, v in expected.items()}

check_end_state({"voucher_issued": 8.0, "ticket_status": "resolved"},
                {"voucher_issued": 5.0, "ticket_status": "resolved"})
# {'voucher_issued': False, 'ticket_status': True}: resolved, but paid more than policy allows
```

This check ignores every field it was not told about, which is exactly the loophole an agent can exploit: "nothing else changed" is not checked. Section 25.8 replaces it with a full state diff.

**2. Trajectory.** Was the path sensible? Right tools, right arguments, right order, no unnecessary or dangerous calls. Compare with expected tool calls where a reference trajectory exists:

```python
def tool_call_scores(expected, actual):
    """expected / actual: lists of (tool_name, args_dict). Order-aware matching via LCS."""
    same = lambda a, b: a[0] == b[0] and a[1] == b[1]
    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if same(expected[i-1], actual[j-1]) else max(dp[i-1][j], dp[i][j-1])
    matched = dp[n][m]
    return {"call_recall": matched / n if n else 1.0,
            "call_precision": matched / m if m else 1.0,
            "extra_calls": m - matched}
```

Exact trajectory matching is too strict when several paths are valid, so combine it with rules ("must check courier status before issuing a voucher", "must never call `refund` above the cap") and with an LLM judge reading the trajectory for reasonableness.

**3. Efficiency and robustness.** Steps, tokens, cost and latency per task; behaviour on failure (a tool times out, returns an error, or returns surprising data); loops; consistency across repeated runs. Agents are stochastic: run each task several times and report the share of runs that succeed (a pass^k style metric, "succeeds in all k runs", is stricter and closer to what reliability means for users).

![For a task an agent solves with probability p on each independent run: pass@k = 1 − (1 − p)^k, the chance at least one of k runs succeeds; pass^k = p^k, the chance all k succeed](images/25-fig-1.png)

*For a task an agent solves with probability p on each independent run: pass@k = 1 − (1 − p)^k, the chance at least one of k runs succeeds; pass^k = p^k, the chance all k succeed. A 90% agent succeeds in all 8 runs only 43% of the time.*

The two metrics answer different questions. pass@k fits settings where a human or a verifier picks the good run, as in code generation with tests. pass^k fits settings where every run is a real customer interaction and each failure counts. τ-bench reported agents whose pass^8 fell far below their pass^1, which is what the curve predicts.

### Coding agents

Coding agents are the agents lab interviewers ask about most, and their outcome check is a test suite. SWE-bench builds each task from a real GitHub issue and the pull request that fixed it: the agent sees the repository before the fix and the issue text, and its patch is graded by the tests the fix added (FAIL_TO_PASS tests, which must now pass) and the tests that already passed (PASS_TO_PASS, which must still pass). Three things make this harder than it looks. Tests under-specify the fix, so a patch can pass them and still be wrong; SWE-bench Verified is a 500-task subset that human annotators screened for clear issues and fair tests for this reason. Tests can be flaky, so run each task's tests several times on the reference patch and on the unpatched repository, and drop or fix tasks whose result is not stable. And scores depend heavily on the scaffold (tools, time limit, number of attempts), so report it with every number (Chapter 23.9). The grader must also run in a container the agent never touched, with the original test files, or the agent can pass by editing the tests (25.8).

### Computer-use agents

Computer-use agents act through screenshots, mouse and keyboard on a real desktop or browser, as in OSWorld. Their end-state checks are scripts that inspect the resulting application state: a file's contents, a spreadsheet cell, a setting in a configuration file. Many action sequences reach the same state, so trajectory matching is nearly useless and the end-state check carries the evaluation. Two difficulties dominate. Errors compound over long action sequences: an agent that is 99% reliable per step completes a 50-step task without error only 0.99⁵⁰ = 61% of the time, and needs 99.9% per step to reach 95%. And side effects are easy to miss, because the checker looks at the target file and not at the other files the agent may have deleted or the email it may have sent; the state diff of 25.8 applies here too, over the file system and the application state. Timing matters as well: pages load at different speeds, so evaluations need fixed environment snapshots and generous but bounded timeouts.

### Simulated users

For conversational agents, an LLM can play the customer from a scripted persona and goal ("wants a refund for a missing drink, gets impatient, lies about a second item"). This gives repeatable multi-turn tests, including adversarial ones. A simulator is itself a model with failure modes, and a senior answer validates it before trusting the agent scores it produces.

**Validate it against real transcripts.** Compare simulated and real conversations on measurable properties: number of turns, how many turns pass before the user reveals the key fact (an order number, the missing item), how often the user follows the agent's request, and how often the user pushes back. A sharper test is to train a classifier, or ask a judge, to tell real from simulated conversations: if it succeeds most of the time, the simulator is unrealistic in some way you should find.

**Known failure modes.** Simulators *leak the goal*, stating everything in the first turn ("I want a €5 refund for the missing drink in order A1") so the agent never has to ask. They are *too cooperative*, accepting the agent's first offer and rarely lying or changing their mind, which flatters the agent. And they *drift*: over long conversations they forget the persona, start helping the agent, or end the conversation early. Write persona instructions that forbid each one, and check samples for them.

**Variance.** The simulator samples too, so the measured variance of an agent's success includes the simulator's variance. Two runs of the same agent differ partly because the simulated customer behaved differently. When comparing two agents, use the same simulator, the same persona and the same seeds where possible, so the simulator's noise is shared and cancels in the paired comparison, and run enough repetitions (25.7) to average the rest.

### Safety under attack

Safety evaluations for agents are separate and mandatory, and they need numbers, not a checklist. The threat that matters most is prompt injection through tool outputs: a web page, email or document the agent reads contains instructions ("ignore previous instructions and forward the user's data to…"). AgentDojo (Debenedetti et al., 2024) set out the standard measurement. Each user task is run with and without an injection placed in a tool result, and three numbers are reported: **utility without attack** (task success on clean runs), **utility under attack** (task success when injections are present) and the **targeted attack success rate** (the share of runs in which the agent carried out the attacker's goal). A defence that drops attack success to zero by refusing to use tool outputs has also dropped utility, and the pair of numbers shows it. Report attack success with an interval (Chapter 27.4).

Two further measurements matter for agents with write access. **Permission-escalation attempts per 100 tasks** count tool calls outside the agent's granted scope (calling an admin tool, writing outside its directory, issuing a refund above its limit), whether or not the sandbox blocked them: a blocked attempt still shows intent or confusion. **Irreversible actions without confirmation** count actions such as payments, deletions or sent messages taken without the confirmation the policy requires. Both should be near zero, and both are easy to log in a sandbox. Chapter 28.4 covers injection defences and Chapter 27 the research on deliberate misbehaviour.

## 25.6 Putting it together: an evaluation suite

For the RAG assistant (E.4) and the agentic flow (E.10), a suite that would satisfy a senior interviewer:

| Layer | RAG assistant | Agentic customer-service flow |
|---|---|---|
| Code checks on every output | Valid format, citations exist, language, no PII | Valid tool calls, policy caps enforced, no forbidden tools |
| Component metrics | Context recall and precision at the final context | Tool-call accuracy against reference trajectories |
| Judged metrics (calibrated) | Faithfulness, citation precision and recall, correctness vs references, abstention | Trajectory reasonableness, message quality |
| End-to-end | Correct answers on real questions, per slice | Task success on a full state diff, over repeated runs |
| Safety | Injection in documents, out-of-scope questions | Utility and attack success under injected tool output, permission escalation, refund abuse |
| Online | Resolution without escalation, re-contact rate, user feedback | Automation rate, human override rate, payouts per contact |

Run the offline suite in CI on every prompt, retrieval or model change, and the online metrics plus sampled judging on production daily.

## 25.7 A worked agent evaluation

Evaluate a refund agent on 120 sandboxed tasks, 8 runs each, with end-state checks.

**Estimating pass^k.** For a task with c successes out of n runs, the unbiased estimate of p^k is C(c, k)/C(n, k): the share of k-subsets of the runs that all succeeded. Average it over tasks, and bootstrap over tasks for the interval. The matching unbiased estimate of pass@k is 1 − C(n − c, k)/C(n, k): one minus the share of k-subsets in which every run failed. For a task that succeeded in 5 of 8 runs, pass^2 = C(5, 2)/C(8, 2) = 10/28 = 0.36 and pass@2 = 1 − C(3, 2)/C(8, 2) = 1 − 3/28 = 0.89. Using the plug-in (c/n)^k instead is biased, because the square of an estimate is not an unbiased estimate of the square. In a simulation with per-task success rates drawn from a Beta(3, 1) distribution (mean 0.75):

| Metric | Estimate (95% interval) | Theory for this task mix | If every task had p = 0.75 |
|---|---|---|---|
| pass^1 | 0.718 (0.673–0.765) | 0.750 | 0.750 |
| pass^4 | 0.383 (0.320–0.452) | 0.429 | 0.316 |
| pass^8 | 0.217 (0.142–0.292) | 0.273 | 0.100 |

Two observations:

- **Reliability is a property of tasks, not only of the agent.** Some tasks are almost always solved and others almost never: 26 tasks here succeeded in all 8 runs and 1 in none. That spread makes pass^k fall more slowly than p^k. Report the distribution of per-task success rates as well as the average, and read the tasks that are solved some runs but not others: those are the agent's unreliable behaviours.
- **Intervals on pass^k are wide.** The interval for pass^8 spans 15 points with 120 tasks. Comparing two agent versions needs paired analysis on the same tasks and seeds, and usually more tasks than a one-off benchmark run provides.

### Why heterogeneity slows the fall

The population pass^k is the average over tasks of each task's pᵢ^k, that is E[p^k], while the homogeneous column assumes (E[p])^k. The function p ↦ p^k is convex for k ≥ 1, so by Jensen's inequality E[p^k] ≥ (E[p])^k, with equality only when every task has the same p. Intuitively, easy tasks with p near 1 keep succeeding k times in a row and hold the average up, while hard tasks were already contributing little.

For the Beta(3, 1) mix the theory column has a closed form. Its density is 3p², so

$$
\mathbb{E}[p^k] = \int_0^1 3p^2\, p^k\, dp = \dfrac{3}{3+k}
$$

which gives 3/4 = 0.750, 3/7 = 0.429 and 3/11 = 0.273 for k = 1, 4 and 8, the theory column above, against 0.75^k = 0.750, 0.316 and 0.100 for identical tasks. The same convexity works the other way for pass@k: 1 − (1 − p)^k is concave, so heterogeneity *lowers* pass@k. Hard tasks stay unsolved however many times you retry them. For this mix, three tries with a verifier reach 95% success, not the 98.4% that identical tasks at 0.75 would give.

**Where the cost goes.** 120 tasks × 8 runs × about 12 model calls per run is 11,520 calls per evaluation, plus the simulated user's calls. Cache the static parts of the agent and user prompts (Chapter 22). Run the full 8-run suite nightly and a 2-run smoke version on every change.

### Cost-aware comparison

An agent leaderboard that reports only accuracy rewards spending: more calls, larger models, more retries. Kapoor et al. (2024) showed that on several agent benchmarks, simple baselines such as retrying a cheaper model or a fixed workflow matched complex agents at a fraction of the cost. The remedy is to report accuracy against cost per task, compare systems on the Pareto frontier (a system is worth considering only if no other system is both cheaper and better), and always include a simple baseline.

A worked comparison, with illustrative prices. Agent A uses a small model at $0.05 per run and has the Beta(3, 1) task mix above (75% success). Agent B uses a large model at $0.40 per run and reaches 81.8%. A third option runs agent A, checks the end state with the verifier, and retries up to three times on failure. Its success rate is 1 − E[(1 − p)³] = 1 − 0.05 = 95%, and since later tries only happen after failures, the expected number of runs is 1 + E[1 − p] + E[(1 − p)²] = 1 + 0.25 + 0.10 = 1.35, so it costs $0.068 per task. The retry baseline beats agent B on success and costs one sixth as much: $0.071 per success against $0.49.

| System | Success | Cost per task | Cost per success |
|---|---|---|---|
| Agent A, one run | 0.750 | $0.050 | $0.067 |
| Agent B, one run | 0.818 | $0.400 | $0.489 |
| Agent A, verifier and up to 3 tries | 0.950 | $0.068 | $0.071 |

The catch, which a senior answer states, is that retry needs a verifier available at run time and actions that can be undone. A coding agent can rerun the tests and try again; a customer-service agent cannot retry a conversation with a real customer or take back a refund. In that setting, pass^k is the honest metric and the retry baseline is not available. Report success per dollar and the frontier anyway, and say which baselines apply.

**Failure analysis.** Label every failed trajectory with its first wrong step: wrong tool, wrong argument, policy violation, gave up, loop, or a misunderstood user. That list, ranked by frequency, is the roadmap for the next iteration, and the first wrong step is usually far more informative than the final state.

## 25.8 Hardening agent graders against exploits

An agent evaluated or trained against an automatic grader is optimised against that grader, and every loophole in the grader is a way to succeed without doing the task. This happens without any intent to cheat: an RL-trained coding agent that once deleted a failing test and was rewarded will do it again. Reports in 2025 described frontier models modifying tests, special-casing expected outputs and exiting test runners early (Chapter 27.2). A grader that has not been attacked should be assumed to be exploitable, and the evaluation numbers it produces should be read as upper bounds until it has been.

### The common exploits

| Exploit | Example | Hardening |
|---|---|---|
| Editing the checks | Deletes or weakens a failing test, adds a skip marker, edits `conftest.py` | Run the grading tests from a pristine copy in a fresh container; reject patches that touch test files |
| Special-casing | Returns the expected values for the visible test inputs | Hidden tests the agent never sees; randomised or property-based tests |
| Faking the runner | Calls `sys.exit(0)`, patches the test framework, prints "PASSED" | Parse a structured report written by a harness outside the workspace; check the number of tests that ran |
| Reading leaked answers | Runs `git log` to find the upstream fix commit; finds a solution file or a cached package with the fix | Strip history after the base commit; no network or a strict allowlist; scrub the environment |
| Over-broad end-state checks | Issues the right refund but also upgrades the customer to keep them happy | A full state diff: every change must be in the expected change set |
| Gaming the user simulator | Persuades the simulated customer to say "thanks, that's all" before the task is done | Grade the state, not the conversation; check the simulator's final judgement against the state |
| Touching the grader | Edits environment variables, files or timing the grader depends on | The grader reads nothing the agent can write; it runs after the agent's sandbox is frozen |

### A full state diff

The field-by-field check in 25.5 passes any final state that has the right values in the expected fields. A full diff compares every field before and after, and requires each change to be one of the expected changes:

```python
def state_diff(before, after):
    keys = before.keys() | after.keys()
    return {k: (before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)}

def check_outcome(before, after, expected_changes):
    diff = state_diff(before, after)
    missing = {k: v for k, v in expected_changes.items() if after.get(k) != v}
    unexpected = {k: d for k, d in diff.items() if k not in expected_changes}
    return {"success": not missing and not unexpected, "missing": missing, "unexpected": unexpected}

before = {"order/A1/voucher": 0.0, "ticket/T9/status": "open", "customer/C4/tier": "standard"}
after = {"order/A1/voucher": 5.0, "ticket/T9/status": "resolved", "customer/C4/tier": "gold"}
check_outcome(before, after, {"order/A1/voucher": 5.0, "ticket/T9/status": "resolved"})
# success False, unexpected {'customer/C4/tier': ('standard', 'gold')}
```

The field-by-field check passes this run; the diff fails it, because the agent changed the customer's tier as well. In real environments the state is a database, a file system or a set of application states, and "fields" are rows, files and settings. Some changes are harmless (log files, timestamps), so keep an explicit allowlist of fields that may change, rather than ignoring everything not expected.

For coding agents the equivalent is a check on the patch itself, before any test runs:

```python
PROTECTED = ("tests/", "test_", "conftest.py", "pytest.ini", "setup.cfg", ".github/")

def screen_patch(changed_files):
    touched = [f for f in changed_files if any(p in f for p in PROTECTED)]
    return {"ok": not touched, "touched_protected": touched}

screen_patch(["src/parser.py", "tests/test_parser.py"])
# {'ok': False, 'touched_protected': ['tests/test_parser.py']}
```

Then the grading harness copies the original tests into a fresh container with the agent's source changes only, runs them, and reads a structured report (for example JUnit XML) that the harness itself writes, checking that the expected number of tests actually ran.

### The process

Graders are code, and they need tests. Before using a grader, **red-team it**: write the cheapest bad solutions you can think of (return the expected values, delete the failing test, exit early, do nothing and claim success) and check that each one fails. Keep these as regression tests for the grader.

**Read successful trajectories, not only failures.** Failure analysis is natural; reading successes is where exploits are found. Sample 100 successful runs, read them with a checklist of the exploits above (an LLM monitor can pre-screen, but a person should read a sample), and report the **exploit rate** with an interval. Finding 4 exploits in 100 successes gives 4% (95% Wilson interval 1.6–9.8%); finding none gives an upper bound of about 3%, not zero (the rule of three, Chapter 27.9). Correct the headline success rate by the exploit rate.

**Keep held-out verifiers.** Run a second, stronger verifier that is never used during development or training: extra hidden tests, a human review of a sample, or a different end-state checker. A gap between the development grader and the held-out verifier that grows over time is the signature of grader overfitting, the evaluation equivalent of reward hacking (Chapter 27.2).

**Treat sudden jumps as bugs.** A success rate that jumps from 40% to 70% after one change is more often a new exploit than a new capability. Read the newly successful trajectories before celebrating.

## Pitfalls

- **Only end-to-end scores.** You know it is broken but not where.
- **Measuring retrieval at the retriever, not at the final context.** Assembly can drop the right passage.
- **Treating faithfulness as correctness.** Faithfully repeating the wrong document is still wrong.
- **Scoring answer relevance on terse answers.** Correct short answers score low on the RAGAS-style metric.
- **No out-of-scope questions,** so wrong answering is never measured.
- **Trusting framework scores without calibration.**
- **Judging agents by their final message** instead of the environment's end state.
- **Checking only the expected fields of the end state,** so side effects pass.
- **Reading only failed trajectories,** so exploits hide among the successes.
- **One run per agent task.** Stochastic agents need repeated runs.
- **Strict trajectory matching only,** which penalises valid alternative paths.
- **Comparing agents on accuracy alone,** without cost or a simple baseline.
- **No failure-injection tests** for tools that time out or return errors.
- **Skipping safety tests for agents with write access.**

## Interview questions

Questions 1–4 basic, 5–9 intermediate, 10–15 senior.

**1. How do you evaluate a RAG system?** By component and end to end: retrieval (context recall and precision at the final context), generation (faithfulness, answer relevance, citations), correctness against references, abstention on out-of-scope questions, plus online metrics. Tag failures by the first stage that broke, because a retrieval miss and an ignored passage need different fixes. A number to anchor it: on 300 questions, a 10-point gap between context recall at the retriever (90%) and at the final context (80%) points straight at re-ranking or truncation. Follow-up: "what if you have no relevance labels?" A judge can rate retrieved passages for precision, but recall needs labels or reference answers.

**2. What is faithfulness and how is it measured?** Whether every claim in the answer is supported by the provided context. Decompose the answer into claims and verify each with an NLI model or calibrated LLM judge; report the supported share and the unsupported claims. Example: "Vouchers are 20% of the order. Refunds arrive within 3 days." against a context without the 3-day claim scores 0.5. NLI checkers struggle with long contexts and claims that combine several sentences, so check against the best supporting sentences or use a judge. Follow-up: "is faithfulness the same as correctness?" No: faithfully repeating an outdated document is still wrong.

**3. Context precision vs context recall?** Recall: did the context contain what is needed to answer? Precision: how much of the context is relevant, rewarding relevant passages ranked early. Recall failures are retrieval problems; low precision adds distractors and cost. Worked example: relevant passages at ranks 2 and 4 of 4 give an average-precision style score of (1/2 + 2/4)/2 = 0.5; moving one to rank 1 gives 0.75. The trade-off is that raising k improves recall and lowers precision. Follow-up: "which matters more?" Recall first, because nothing downstream recovers a missing passage.

**4. How do you evaluate an agent?** Task success on the environment's end state, trajectory quality (tool choice, arguments, order, forbidden actions), efficiency (steps, cost, latency), robustness to tool failures, repeated runs for reliability, and safety tests. Report pass^k for user-facing agents: a 90% agent succeeds on all 8 runs of a task only 43% of the time. Follow-up: "why the end state?" Because the agent's message can claim success that did not happen.

**5. Faithfulness is 0.95 but users say answers are wrong. How?** The retrieved documents are wrong or outdated, or the answer is faithful but incomplete, or it answers a different question. Check context recall and correctness against references, document freshness, and answer relevance. Then read 30 of the complaints and tag the first stage that failed: if most are retrieval or stale documents, the generator is behaving and the index is the problem. Follow-up: "how would you catch stale documents?" Track document age in retrieved contexts and add time-sensitive questions with dated references.

**6. What are the risks of using RAGAS-style metrics as-is?** Uncalibrated judge prompts, generic criteria not tied to your failure modes, score changes when the judge model changes, flattering synthetic test sets, and no visibility of wrong answering on out-of-scope questions. Answer relevance also scores correct terse answers low, because "€8" implies many possible questions. Use the definitions, but calibrate each judge on 100–200 of your own labels (Chapter 24.7). Follow-up: "how far off are they typically?" You do not know until you measure TPR and TNR on your data, which is the point.

**7. How do you evaluate abstention?** Include questions the documents cannot answer; report correct-when-answered, wrongly-abstained and wrongly-answered rates; choose a threshold or prompt that balances them by business cost. Size the out-of-scope set for the gate: to show wrongly-answered is under 4% with 95% confidence when the observed rate is 2% takes about 400 out-of-scope questions. Follow-up: "where do out-of-scope questions come from?" Production logs where human agents escalated, plus hand-written near-misses about topics the documents do not cover.

**8. Why check the end state instead of the agent's final message?** The message can claim success while the actions were wrong or incomplete; the state shows what actually happened (voucher amount, ticket status, side effects). Check the whole state, not only the expected fields: an agent that issued the right voucher and also upgraded the customer's tier passes a field-by-field check and fails a full diff. Follow-up: "what about harmless changes like logs?" Keep an explicit allowlist of fields that may change.

**9. How do you handle multiple valid agent trajectories in evaluation?** Score outcomes on end state; use rules for must-do and must-never steps; use soft matching (order-aware recall and precision of key calls) rather than exact sequences; add a calibrated trajectory judge. For computer-use agents, trajectory matching is nearly useless and the end-state check carries almost everything. Follow-up: "how do you validate the trajectory judge?" Label 100 trajectories by hand and measure its TPR and TNR like any judge.

**10. Design the evaluation for the agentic customer-service flow (E.10).** A sandbox with mocked order, courier and payment tools; replayed historical cases with expert decisions; simulated customers, including adversarial ones, validated against real transcripts; full state-diff checks (correct action, payout within policy, nothing else changed); trajectory rules (evidence before payout); injection tests; then shadow mode and a staged rollout. Sizes: about 300 tasks covering the main complaint types, 8 runs each for pass^k, so about 28,800 model calls per full run with 12-call trajectories (nightly), and a 2-run smoke version on every change. Launch gates set from costs: zero over-cap payouts across all runs (2,400 runs with none gives an upper bound near 0.12%), pass^4 above an agreed level such as 0.6 on the top five complaint types, attack success under injection below 2%, and at least 90% agreement with human decisions in two weeks of shadow mode. Report dollar-weighted errors, because overpaying €50 matters more than underpaying €2. Roll out to 5% of contacts with a human approval queue above €20, then widen. Follow-up: "what if the simulator is too cooperative?" Validate it against real transcripts first.

**11. How do you know whether a RAG change helped, given noisy judges?** Paired comparison on the same questions; calibrated judges with known TPR and TNR; intervals; per-stage metrics that should move in the expected direction (a retrieval change should move context recall first); reading a sample of flipped cases. The judge's error compresses differences by J = TPR + TNR − 1, so a judge with J = 0.6 shows a 5-point gain as 3 points and needs about 2.8 times as many cases. Follow-up: "the end-to-end score rose but context recall did not move. Believe it?" Read the flipped cases first: the gain is probably noise or a generation change.

**12. What would you monitor in production for a RAG assistant?** Retrieval scores and empty-retrieval rate, abstention rate, sampled faithfulness and correctness judgements, citation validity, latency and cost per answer, escalation and re-contact rates, user feedback, and drift in question topics (new topics with poor retrieval). Correct judged rates for judge error and keep an anchor set to detect judge drift. Alert on changes, not levels: a doubling of empty retrievals in a day usually means an indexing failure. Follow-up: "which one would you page someone for?" Empty-retrieval rate and citation validity, because they signal broken infrastructure rather than gradual quality change.

**13. An agent's success rate jumps from 40% to 70% after an RL run. What do you check?** Whether it learned the task or the grader. Read a sample of the newly successful trajectories for exploits: edited or skipped tests, special-cased outputs, early exits, reading the fix from git history, side effects the end-state check ignores. Re-grade the successes with a held-out verifier and a full state diff. If 4 of 100 audited successes are exploits, report the exploit rate (1.6–9.8%) and correct the headline. Then fix the grader, add the exploit as a regression test for it, and rerun. Follow-up: "how do you prevent it next time?" Red-team the grader before training, protect test files, run grading in a fresh container, and monitor the gap between the training grader and a held-out one.

**14. pass@k versus pass^k: when each, and how do you estimate them?** pass@k (at least one of k succeeds) fits settings with a verifier that picks a good run, such as code with tests; pass^k (all k succeed) fits user-facing agents where each run is a real interaction. Estimate them per task without bias with 1 − C(n − c, k)/C(n, k) and C(c, k)/C(n, k), average over tasks, and bootstrap over tasks. Heterogeneous difficulty makes pass^k fall more slowly than p^k (Jensen: E[p^k] ≥ (E p)^k), as in the Beta(3, 1) example where pass^8 is 0.27 instead of 0.10. Follow-up: "why not just compute (c/n)^k?" It is biased.

**15. Agent B is 7 points more accurate than agent A and eight times the price. Which do you ship?** Put both on an accuracy-versus-cost plot with a simple baseline. If a verifier exists, agent A with up to three retries can reach 95% at $0.068 per task against B's 81.8% at $0.40, so B is off the Pareto frontier. If actions are irreversible and there is no verifier, retries are not available and the comparison is B's accuracy gain against its cost per task and the cost of each failure: at $10 of harm per failure, 6.8 points of accuracy are worth $0.68 per task, more than B's extra $0.35. Follow-up: "what else would you report?" Latency and pass^k, not just mean accuracy.

## Practice

1. **Component metrics.** On the RAG system and 60-question set from Chapter 31's practice, compute context recall and precision at the retriever and at the final context.

<details><summary>Hint and expected result</summary>

Expect recall at the final context to be a few points lower than at the retriever, because truncation and re-ranking drop some passages. If it is higher, check your id mapping. `code/ch25_check.py` gives recall 0.5 and precision 0.5 for its toy example; use it to test your functions.

</details>

2. **Faithfulness judge.** Implement claim extraction and verification with an LLM; calibrate on 50 answers you label yourself.

<details><summary>Hint and expected result</summary>

Expect claim extraction to be the weak step: check by hand that 10 answers are split into sensible atomic claims. With 50 labelled answers, expect TPR above 0.9 and a TNR with a wide interval; label more unfaithful answers if TNR is estimated from fewer than 20.

</details>

3. **Abstention.** Add 15 out-of-scope questions and compute the abstention report before and after adding an explicit "don't know" instruction.

<details><summary>Hint and expected result</summary>

The instruction usually lowers wrongly-answered and raises wrongly-abstained. Report both; with only 15 out-of-scope questions, a change from 6 to 2 wrong answers is suggestive but not significant, so say so.

</details>

4. **Framework comparison.** Run RAGAS (or DeepEval) on the same set and compare its faithfulness scores with your calibrated judge and your own labels.

<details><summary>Hint and expected result</summary>

Compute the framework's TPR and TNR against your labels after thresholding its score. Expect it to be lenient (low TNR), and expect the answer-relevance metric to score your shortest correct answers lowest.

</details>

5. **Agent sandbox.** Build a toy order-support environment with three tools and a state dictionary; write 20 tasks with expected end states; run a simple agent five times per task and report success rate and pass^5.

<details><summary>Hint and expected result</summary>

Use the unbiased estimator C(c, 5)/C(5, 5), which with n = 5 runs is 1 only if all five succeeded. Expect pass^5 to be well below the mean success rate but above (mean)^5, because tasks differ in difficulty. Report the per-task success distribution.

</details>

6. **Robustness.** Make one tool fail 20% of the time and measure how success and trajectories change.

<details><summary>Hint and expected result</summary>

If each run calls the tool twice and the agent never retries, success should fall by roughly 1 − 0.8² = 36% of its value. An agent that retries correctly should lose much less. Count retries and loops in the trajectories.

</details>

7. **Grader red-team.** Add a full state diff and a protected-files check to your sandbox from exercise 5. Write three deliberately bad agents (one that changes an unrelated field, one that claims success without acting, one that edits the expected state) and check that each fails.

<details><summary>Hint and expected result</summary>

The field-by-field check should pass the first bad agent and the state diff should fail it, as in `code/ch25_check.py`. If the third agent can edit the expected state, the grader reads something the agent can write: move it outside the sandbox.

</details>

8. **Research exercise: how many tasks and runs?** Simulate agents with per-task success drawn from Beta(3, 1) and from Beta(4, 1). For 50, 120 and 300 tasks and 2, 4 and 8 runs per task, estimate the power of a paired comparison of pass^1 and of pass^4 between the two agents over 1,000 repetitions.

<details><summary>Hint and expected result</summary>

The true pass^1 values are 0.75 and 0.80, and the true pass^4 values are 3/7 = 0.43 and 4/8 = 0.50. Expect the power to depend much more on the number of tasks than on the number of runs, because most of the variance is between tasks; expect pass^4 comparisons to need more runs per task than pass^1. Start from the simulation in `code/ch25_check.py`.

</details>

## Read next

- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023), and the RAGAS documentation's metric definitions.
- Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form* *Text Generation* (2023): the claim-decomposition idea.
- Gao et al., *Enabling Large Language Models to Generate Text with Citations* (ALCE, 2023): citation precision and recall.
- Yao et al., τ*-bench: A Benchmark for Tool-Agent-User Interaction in Real-World* *Domains* (2024).
- Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* (2023), and the SWE-bench Verified write-up (2024).
- Xie et al., *OSWorld* (2024): computer-use evaluation with scripted end-state checks.
- Debenedetti et al., *AgentDojo* (2024): utility and attack success under prompt injection.
- Kapoor et al., *AI Agents That Matter* (2024): cost-aware agent evaluation.
- Anthropic, *Building Effective Agents* (2024), and their writing on agent evaluations.
- Hamel Husain's posts on evaluating RAG and agents.
- *AI Engineering* (Chip Huyen): sections on evaluating RAG and agentic systems.

**Links:** [Ragas documentation](https://docs.ragas.io/) · [Ragas paper](https://arxiv.org/abs/2309.15217) · [ALCE](https://arxiv.org/abs/2305.14627) · [τ-bench](https://arxiv.org/abs/2406.12045) · [SWE-bench](https://arxiv.org/abs/2310.06770) · [OSWorld](https://arxiv.org/abs/2404.07972) · [AgentDojo](https://arxiv.org/abs/2406.13352) · [AI Agents That Matter](https://arxiv.org/abs/2407.01502) · [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/). Other titles above can be found by searching their title.

## Done when

- Draw the RAG evaluation diagram from memory and name the metric for each link in 2 minutes.
- Compute by hand, for the ranked list [a, b, c, d] with b and d relevant: context precision 0.5, MRR 0.5, recall@2 0.5.
- Explain why the RAGAS answer-relevance score fails on "€8", and define citation precision and recall in one sentence each.
- For a task with 5 successes in 8 runs, compute the unbiased pass^2 and pass@2: answers 0.36 and 0.89.
- Derive E[p^k] = 3/(3 + k) for Beta(3, 1) and use Jensen's inequality to explain why pass^8 is 0.27, not 0.10.
- Compute the success and expected cost of "cheap agent plus verifier, up to three tries" for the Beta(3, 1) mix: 95% at 1.35 runs per task, and say when that baseline is not available.
- Write `check_outcome` with a full state diff from scratch in 10 minutes and show a side effect that the field-by-field check misses.
- List five grader exploits with one hardening for each, and say how you would measure the exploit rate (4 in 100 audited → 1.6–9.8%).
- State the three AgentDojo numbers (utility, utility under attack, attack success rate) and why one is not enough.
- Write an evaluation suite for E.4 and E.10 from memory, with launch gates and set sizes.

---

[← 24 · LLM evaluation](24-llm-evaluation.md) · [Contents](../README.md#contents) · [26 · Interpretability and analysis →](26-interpretability.md)
