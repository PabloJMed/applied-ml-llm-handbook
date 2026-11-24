# 39 · Thinking like a senior

Part X · Research craft and interviews. No prerequisites; draws on the system-design and experiment chapters for examples. Chapter 38 covers how to show these qualities in behavioural interviews.

## Why it matters

Technical depth gets you into the room; judgement is what interviewers score for seniority. Across every round (coding, ML depth, system design, behavioural) they listen for the same signals: do you start from the problem rather than the model, do you name trade-offs, do you know when to stop, can you say "I'm not sure" usefully, and do other people get better around you? This chapter turns those signals into habits and phrases you can practise. It is short on purpose; the practice is in using it in every other chapter.

The chapter covers the habits that distinguish senior work, each with concrete examples. It starts with reasoning about trade-offs (39.1) and product sense (39.2), then scoping and saying no (39.3), communicating uncertainty (39.4) and leading without authority (39.5). Sections 39.6 and 39.7 turn these into how you speak in an interview, and 39.8 covers what seniority looks like in research specifically. Section 39.9 works through a product scoping conversation, and 39.10 its research counterpart: a lead asks you to test a method for the next pretraining run in three weeks. Section 39.11 covers triage under pressure, deciding whether to stop a costly run and which hypothesis to check first, and 39.12 ends with a rubric for placing yourself at mid, senior or staff level on five axes, so you know which gap to work on.

What these habits share is a focus on decisions rather than tasks. A junior engineer is given a well-defined problem and solves it; a senior one works out which problem is worth solving, what would be good enough, and what to tell the people who must decide. Interviews for senior roles test exactly that shift.

## Key concepts

- **Trade-off**: every option is better on some axes and worse on others; seniority is naming the axes and choosing for the situation.
- **One-way vs two-way doors**: decisions that are hard to reverse deserve slow, careful thought; easily reversed ones should be made fast. The idea comes from Amazon's shareholder letters; Chapter 39.1 applies it.
- **Baseline**: the simplest reasonable solution (a rule, a heuristic, a popularity ranker) that a model must beat to be worth building. Google's [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) make the case at length.
- **Guardrail metric**: a metric that must not get worse when the main metric improves (Chapter 2).
- **Confidence interval**: a range that expresses uncertainty about an estimate. [Explainer](https://en.wikipedia.org/wiki/Confidence_interval).
- **Disagree and commit**: argue your case once, clearly, then fully support the decision. [Background](https://en.wikipedia.org/wiki/Disagree_and_commit).
- **Influence without authority**: changing what others do through writing, data and trust rather than reporting lines. [StaffEng](https://staffeng.com/) covers it well.

## 39.1 Trade-off reasoning

Every real decision trades something. Name the axes before choosing:

- **Quality vs latency vs cost** (Chapter 21): a cross-encoder reranker adds accuracy and 80 ms.
- **Speed to ship vs long-term maintainability**: a prompt on an API this week vs a fine-tuned model in two months.
- **Precision vs recall**, priced by the cost of each error (Chapters 4 and 28).
- **Build vs buy vs reuse**: a vendor, an internal platform, or your own.
- **Reversible vs irreversible**: move fast on two-way doors (a feature flag), slow down on one-way doors (a data schema others will depend on, a public commitment to merchants).

A pattern that works in any interview answer: *"There are two reasonable options, A and B.* *A is better on x, B on y. Given [the constraint that matters most here], I'd pick A, and I'd* *revisit if [signal] changed."* That last clause shows you know the decision depends on facts that might move.

Quantify when you can, even roughly: "about 2M calls a month, so even a cent per call is $20k; worth a week of work to cut". Rough numbers stated with their assumptions beat vague adjectives.

## 39.2 Product sense

Start from the user and the business metric, then work back to the model.

1. **Who is the user and what is the job?** A customer deciding what to eat in 30 seconds; a merchant updating a menu; a courier deciding whether to accept an order; a support agent resolving a complaint.
2. **What metric moves if this works?** Orders, conversion, basket size, retention, contact rate, delivery time. Name a guardrail too (cancellations, complaints, merchant fairness).
3. **What is the simplest thing that could move it?** Often a rule, a heuristic or better data beats a model. A popularity baseline is the first recommender; a keyword list is the first classifier.
4. **How would you know it worked?** An offline metric that correlates with the online one, then an experiment.
5. **What could go wrong for the user?** Bad recommendations erode trust; a wrong ETA costs more than a vague one.

In a three-sided marketplace (customers, merchants, couriers), every change has three audiences. A ranking change that lifts orders but starves new restaurants hurts supply next quarter. Saying this unprompted is a strong signal.

**The same questions in an LLM lab.** The "user" is often another team and the "product" is the next model. Pretraining, post-training, evaluation, safety and inference teams are the stakeholders; the metric is the next model's quality on the evaluations the lab trusts, its cost to train and serve, and its safety results. The five questions translate directly: who will use this result (the team freezing the data mixture, the release review), what number moves (a target evaluation at fixed compute, a compute multiplier, a jailbreak rate), what is the simplest thing that could move it (a data change before an architecture change), how would you know (a held-out evaluation with intervals, measured at two scales), and what could go wrong (a gain on one benchmark that costs another, or a capability that raises a safety concern). A change that improves maths scores but breaks the inference team's speculative-decoding draft model has two audiences too.

## 39.3 Scoping and saying no

Seniors are trusted with ambiguous problems because they shrink them. Habits:

- Restate the problem in one sentence and get agreement before designing.
- Ask for the decision the work will inform. Analysis without a decision attached is a hobby.
- Cut to a first version that delivers value in weeks: one city, one cuisine, one language, batch before real time.
- Write down what is *out* of scope, and why.
- Say no with a clear reason and an alternative: "We can't do personalised push copy this quarter without a feature store for user history. We can do segment-level copy now, which gets most of the value, and revisit next quarter."
- Kill your own projects when the data says so, and write up what you learned.

**Example.** Asked to "build a food taxonomy for the whole catalogue", a senior first asks which decision needs it (search filters? dietary labels? analytics?), then scopes v1 to the 50 attributes those decisions use, in the two largest languages, for the top cuisines, with a written list of what is deferred. Prepare one story of your own with this shape.

## 39.4 Communicating uncertainty

The goal is to help people decide, not to sound certain.

- **Give ranges and confidence, not points**: "The test shows +1.2% orders, 95% interval roughly +0.3% to +2.1%. I'm confident it's positive; the size is less certain."
- **Separate what you know, what you believe, and what you're guessing.**
- **Lead with the recommendation**, then the evidence, then the caveats that would change it. Executives read the first line.
- **Say "I don't know" with a plan**: "I don't know whether the drop is seasonal; I'll compare with the same period last year by Thursday."
- **Pre-register** what result would make you ship or stop, before the experiment ends.
- In interviews, think aloud and state assumptions ("I'll assume 10k QPS at peak; tell me if that's off"). Interviewers reward visible reasoning even when the final number is wrong.

## 39.5 Leading without authority

Most scientists at mid and senior level do not manage people, yet they are expected to change what other people and teams do. In a lab that means getting a pretraining team to adopt your data finding, an evaluation team to add your test, or a safety team to trust your result; in a product company, getting engineering, product and operations to act on a model. The tools are the same:

- **Write things down.** A one-page design doc or decision memo gets more alignment than meetings. State the problem, options, recommendation, risks, and what you need from whom.
- **Make others successful.** Review code and analyses carefully and kindly; pair with juniors; share reusable tools (an evaluation harness, a prompt-testing script).
- **Build trust before you need it.** Deliver small things reliably; be the person whose numbers hold up.
- **Disagree and commit, in that order.** Make your case with data once, clearly; if the decision goes the other way, commit fully and help measure it.
- **Bring problems with options.** "The labelling budget won't cover all 300 attributes. Three options: … I recommend the second."
- **Work across teams.** Most ML impact depends on other teams: product, engineering and operations in a company; pretraining, post-training, evaluation, inference and safety in a lab. Learn their goals, constraints and vocabulary.

## 39.6 Phrases that signal seniority

Use them when they are true:

- "Before modelling, I'd check whether a simple baseline gets most of the way."
- "What decision will this inform, and by when?"
- "The offline metric I'd trust here is x, because it tracks y online."
- "The risk I'm most worried about is… and here is how I'd detect it."
- "I'd ship this to 5% behind a flag with these guardrails, and roll back if…"
- "I was wrong about x; here's what I changed."
- "That's outside what I've done directly, but here's how I'd approach it."

And habits that signal the opposite: jumping to the newest model, no baseline, no mention of data quality or evaluation, no numbers, blaming other teams, claiming team results as solely yours.

## 39.7 Your story, in one minute

Interviewers often ask "why this role" and "why now". A clear, honest answer helps more than polish. Fill in this template:

- **Why this company:** [one thing about its product or market] + [one thing about the team's technical work you have read or heard about] + [any prior contact with the team or product].
- **What you bring:** [two or three projects in one line each, each with a number: "an LLM classifier that cut labelling cost by x", "a ranking model that lifted CTR by y"] + [one area of depth, such as parameter-efficient fine-tuning].
- **What you are building up:** one honest growth area and what you are doing about it. It shows self-awareness.
- **Why now:** the move makes sense for your career stage (and, if relevant, for practical reasons such as relocation that you have already settled).

Keep it under a minute; let them ask for more.

## 39.8 Research judgement

Seniority in research looks different from seniority in product engineering. The output is not a shipped feature but a decision about what the lab trains next, and the scarce resource is compute and the attention of the people who run large jobs. A senior researcher is the person whose results others can build on without re-checking them, and who spends large runs only when small ones have earned it. The habits below are what that looks like day to day; 39.10 shows them in a worked example.

- **Choosing problems**: you can explain why a direction matters now, what would make it fail, and what you would do instead (Chapter 34).
- **Calibrated claims**: you say "this improves loss by 0.02 ± 0.005 at 1B, untested at 8B", not "this works".
- **Killing your own ideas** early and without drama, when the evidence says so.
- **Multiplying others**: shared infrastructure, evaluation suites, careful reviews of colleagues' experiments.
- **Connecting research to decisions**: knowing which results would change what the lab trains next, and prioritising those.
- **Compute stewardship**: spending large runs only after small-scale evidence justifies them.

## 39.9 A worked scoping conversation

**Request from a product director:** "Can we add an AI assistant to the merchant app by next quarter? Competitors have one."

A junior answer starts estimating how long it takes to build a chatbot. A senior answer turns the request into a decision.

1. **Find the problem.** Ask what merchants would use it for, and what metric should move.
   - Suppose the answer is that support tickets from merchants cost €1.2M a year.
   - 60% of them are about three topics: payouts, menu changes and promotion set-up.
   - The goal is fewer tickets and faster resolution, not "having AI".
2. **Offer options with costs and risks.**

   | Option | What it does | Effort | Main risk |
   |---|---|---|---|
   | A. Better help content and in-app forms for the three topics | Merchants fix things themselves | 4–6 weeks, no ML | Does not handle unusual cases |
   | B. RAG assistant over help articles and policies | Answers questions with citations | 8–10 weeks | Wrong answers about payouts are costly; needs evaluation and guardrails |
   | C. Agent that can change menus and promotions | Takes actions for the merchant | A quarter or more | Permissions, errors that change real listings, abuse |

3. **Recommend and say why.** Do A now, because a large part of the volume is form-shaped. Run B in parallel as a pilot for payouts questions only, where the documentation is good, measured by the deflection rate and the wrong-answer rate on an evaluation set built from past tickets. Defer C until B shows that merchants trust the assistant, and build it with tools that have server-side limits (Chapter 33).
4. **State what would change the plan.**
   - If the pilot's wrong-answer rate on payouts cannot get below an agreed threshold (say 2%), keep B internal as a tool for support agents rather than shipping it to merchants.
   - If A alone removes most of the volume, B's business case shrinks.
5. **Write it down.** A one-page proposal: the problem with its numbers, the options table, the recommendation, the metrics with targets, the evaluation plan, and the date of the first decision review.

The director asked for an assistant and receives a plan that is more likely to reduce tickets by next quarter. It still includes an assistant, scoped to where it can be shown to work. That is what "scoping and saying no" (39.3) looks like in practice: saying yes to the goal, and choosing the means deliberately.

## 39.10 A worked research scoping example

**Request from your lead:** "Can you test whether method X helps, so we can decide whether it goes into the next pretraining run? The recipe is frozen in three weeks." Take X to be a change to the training recipe, such as a new optimiser for the hidden matrices. You have 16 H100s for the three weeks: 16 × 24 × 21 ≈ 8,000 GPU-hours.

A junior answer starts the biggest run that fits. A senior answer starts from the decision: the lab will change a run costing far more than this whole budget only on evidence that the gain is real and survives scale, so the work must produce that evidence or a clear "no" by the freeze date.

1. **Pin down the decision and the bar.** Ask what result would make the lead adopt X: for an optimiser, a loss improvement at matched compute, expressed as a compute multiplier (how much more compute the baseline needs to reach the same loss), that does not shrink with scale, with no new instability. Agree the bar now, for example a multiplier of at least 1.1× at the largest scale tested, and the default if the result is unclear: keep the current recipe, because changing a large run is close to a one-way door.
2. **Size the options.** At about 20 tokens per parameter and 40% MFU, one run costs about 2 GPU-hours at 150M parameters, 13 at 400M, 84 at 1B and 760 at 3B (`code/ch39_check.py`).

   | Option | What it measures | GPU-hours | Share of budget | Main risk |
   |---|---|---|---|---|
   | A. One 1B run per arm | One point estimate | 170 | 2% | No interval, no trend with scale; untuned X loses unfairly |
   | B. 150M/400M/1B ladder, 5-point learning-rate sweep per arm at the two small sizes, 3 seeds per arm at the best rate at every size | The gain, its noise and its trend with scale | 750 | 9% | Says nothing beyond 1B |
   | C. B plus one 3B run per arm | B, plus a check at the largest affordable scale | 2,270 | 28% | 3B runs take about two days each, so they must start by the end of week 2 |
   | D. Put X into the big run without ablations | Nothing until it is too late | 0 | 0% | An expensive run bet on an untested change |

3. **Recommend.** Option C. Run B in the first ten days, tuning each arm's learning rate separately, since an untuned new method against a tuned baseline is the most common way to get a false negative. Start the 3B pair only if the 1B result clears the bar; that is the kill criterion, and it frees most of the budget for other work if X fails early. Measure seed noise at each size before reading any difference.
4. **State what would change the run.** Adopt X if the 3B multiplier is at least 1.1×, consistent with the trend from the ladder, with no loss spikes. Do not adopt it if the gain shrinks with scale (for example a multiplier falling from 1.3× at 150M to 1.05× at 1B), even if it is positive at every size, because the trend predicts no gain at the production scale. If the result is mixed, recommend keeping the recipe and running X as a short continued-training branch of the big run, which costs little and gives a large-scale answer later.
5. **Write it down on day one.** A half-page plan with the decision, the bar, the ladder, the kill criterion and the date of each read-out, sent to the lead and to whoever owns the run's recipe. On the freeze date they receive a recommendation with intervals, not a pile of loss curves.

The lead asked for "a test of X" and receives a decision procedure: it spends just over a quarter of the budget, stops early if X fails, and makes the default explicit. That is research judgement (39.8) applied: a calibrated claim, a kill criterion and compute stewardship in one plan.

## 39.11 Triage: what to check first, when to stop, when to escalate

Seniority shows most clearly when something goes wrong on an expensive system and the clock is running. Three habits separate a senior response from a merely competent one: ordering hypotheses by how cheap they are to check against how likely they are, putting a cost on continuing as well as on stopping, and escalating early with a recommendation rather than late with a problem. Chapter 37 shows the same pattern in its full answers (section F) and its graded answers (section H).

**Order hypotheses by cost of checking times likelihood.** List the plausible causes, estimate roughly how likely each is and how long it takes to check, and start with checks that are both cheap and likely. A five-minute check of a 20% hypothesis comes before a day-long check of a 40% one, because if the cheap one is right you saved a day, and if it is wrong you lost five minutes.

**Price both sides of stop versus continue.** Stopping or rolling back has a visible cost; continuing on a broken trajectory has a larger, hidden one. Write both down in the same units.

**Escalate when the decision exceeds your authority or the cost of delay exceeds the cost of interrupting someone.** Escalate early, with what you know, what you are checking next, and what you recommend.

**Example.** A large pretraining run on 2,048 GPUs is on day 12 of 30. The loss spiked, recovered partly, and now sits 0.03 above the trend line; checkpoints are saved every 4 hours.

- **Costs.** Each hour of the run is 2,048 GPU-hours, about $4,000 at $2 per GPU-hour. Rolling back to the last checkpoint before the spike wastes about 8,200 GPU-hours (about $16,000). The remaining 18 days are about 885,000 GPU-hours ($1.8M), which is what is at risk if the run continues on a damaged trajectory.
- **Hypotheses, cheapest first.** A bad data batch at the spike step (look up the data indices and read the samples: minutes). A hardware fault on one rank (per-rank gradient norms, NaN counts and ECC logs: minutes). Optimisation instability (gradient-norm history, maximum attention and output logits per layer: minutes, already logged). A recent code or configuration change (the deploy log: minutes; replaying the step on a few nodes: hours).
- **Decision.** If the cause is a bad batch or one faulty node, roll back 4 hours, skip the batch or cordon the node, and continue: $16,000 is small against $1.8M. If logits were growing before the spike, roll back and apply a fix (a lower learning rate for the affected parameters, QK-norm or logit clipping) before continuing, because a second spike is likely. If nothing explains it within about two hours, keep the run going while replaying the step offline, since a 0.03 gap often closes, but set a threshold in advance (for example the gap still above 0.02 after 24 hours) that triggers a rollback.
- **Escalation.** Tell the run owner within the hour, with the facts, the checks under way and a recommended action with its cost. A rollback of $16,000 may be yours to make; a change to the recipe in the middle of the run is not.

The same pattern works for a product incident (Chapter 37, question 188) and for a result that suddenly looks too good (Chapter 37, Chain 5).

## 39.12 Where are you? A self-assessment rubric

Levels are judged on behaviour, not on years or on how many algorithms you know. The rubric below shows typical mid, senior and staff behaviour on five axes, with an example of each from research or applied ML. Place yourself honestly on each axis, ideally with a colleague's view as well, then pick the one axis where you sit lowest and find a project in the next quarter that exercises it. Interviewers score the same axes, so the rubric doubles as a check on your stories (Chapter 38.10).

| Axis | Mid | Senior | Staff |
|---|---|---|---|
| **Scope** | Owns a well-defined project inside a team: "run the ablations for this data filter" | Owns a problem end to end, including which projects to run: "decide the data-filtering strategy for the next run" | Owns an area across teams over a year or more: "the lab's data strategy and the tools every team uses for it" |
| **Ambiguity** | Turns a clear question into good experiments | Turns a vague request into a clear question with a decision and a kill criterion (39.10) | Finds the questions nobody has asked yet, and argues for them before they are urgent |
| **Influence** | Teammates trust their results | Other teams adopt their tools or findings without being told to: a shared evaluation harness | Changes how the organisation works: an evaluation standard every release follows |
| **Judgement** | Makes sound choices when given the options | Generates the options, prices them, says what they would not do, and knows when to stop (39.11) | Makes calls with large, hard-to-reverse costs, and is right often enough that others rely on it |
| **Research taste** | Executes a good idea well | Chooses problems that change decisions, and kills their own ideas early | Sets a direction that others build careers on, and spots which fashionable directions will not pay off |

Two cautions. The levels are cumulative: a senior researcher still does everything in the mid column, and does it reliably. And companies weight the axes differently: a research lab weights taste and judgement; a product company weights scope and influence. Read the target's level guide or ask the recruiter, then match your stories to its emphasis.

## Pitfalls

- Treating "senior" as "knows more algorithms".
- Hedging everything so nothing can be wrong; that is not the same as communicating uncertainty.
- Over-scoping in system design rounds: designing the five-year platform instead of the first useful version.
- Talking only about the model in a product question.
- Overclaiming ownership. Say "I" for what you did and "we" for the team, accurately.

## Interview questions

**1. Tell me about a time you decided not to use ML.** Behavioural: use STAR (Chapter 38) and end with what you learned. A strong story names the simpler alternative, the number that made it good enough, and the trigger that would bring ML back. Example shape: "Three rules caught 85% of the cases at 99% precision. A model might have added a few points of recall, but it needed labels, a pipeline, monitoring and retraining. I shipped the rules, and wrote down that if volume doubled or recall fell below 80% we would build the model." Weak answers make the decision sound like laziness or luck. Follow-up: "Did you ever revisit it?" Say what the trigger showed, or that it never fired.

**2. Tell me about a trade-off you made between speed and quality. Would you make it again?** Behavioural: STAR, with both sides of the trade-off quantified (shipped three weeks earlier; 4 points lower accuracy on the long tail), who agreed to it, and how you limited the downside (a flag, a follow-up ticket with a date, monitoring of the weak segment). The second question is the real one: answer it honestly, and "no, I'd do it differently, because…" is a good answer if you can say what you learned. Follow-up: "Who paid the cost of the lower quality?" Name them and say what you did for them.

**3. Your manager asks for a model to predict which restaurants will churn, by next week. What do you do first?** Clarify the decision before the model: who acts on the prediction, how, and with what capacity. If account managers can call 200 restaurants a month, and with 10,000 active restaurants and 2% monthly churn about 200 churn each month, what matters is precision in the top 200, not AUC. Define churn (no orders for 28 days, say), then build a baseline this week: flag restaurants whose orders fell by more than half against the previous four weeks, and measure its precision on last quarter's data. Deliver that with its number and a plan, not a deep model. The senior point: prediction is useless unless calling helps, so propose a holdout in which half the flagged restaurants are not called, to measure whether the intervention reduces churn. Follow-up: "And if the rule is good enough?" Ship the rule and spend the time on the intervention.

**4. You disagree with a product manager about launching a model whose offline metrics look good but whose A/B result is flat. How do you handle it?** Trust the online result over offline metrics, but first check that "flat" is informative. With a 10% baseline conversion rate and 200,000 users per arm, the standard error of the difference is about 0.095 points and the minimum detectable effect at 80% power about 0.27 points, 2.7% relative; if the offline gain predicts +0.5% relative, the test could never have seen it. Then look for reasons (metric mismatch, novelty, opposing segments) and propose an option: a longer or targeted test, or a launch behind guardrails if the change is cheap to reverse. Whose decision it is matters: usually the PM owns the launch, and your job is to make sure it is informed. State your view once, in writing, with the numbers; if the decision goes the other way, commit and help measure it, and escalate only if users are put at risk or a guardrail is broken. Follow-up: "And if the PM wants to ship anyway?" If it is reversible and the guardrails hold, that is a reasonable call; say so.

**5. How do you explain a 95% confidence interval to a non-technical stakeholder?** The strict definition: if we repeated the test many times, intervals built this way would contain the true effect 95% of the time; it is not a 95% probability that this particular interval contains it (Chapter 37, question 9). For a stakeholder: "The test shows +1.2% orders; the effect is very likely between +0.3% and +2.1%. We're confident it's positive; the size is less certain." That plain reading is the approximate Bayesian one with a flat prior. It is close enough for a large A/B test with no strong prior information, and it is what the stakeholder needs to decide, but an interviewer may probe the distinction, so name it. When not to use it: small samples, or when past tests say effects this large are rare; then the winner's curse makes the flat-prior reading overconfident and the estimate should be shrunk. Follow-up: "So what's the probability it's positive?" Only a Bayesian analysis with a stated prior gives that.

**6. What's something you changed your mind about recently, technically?** It tests intellectual honesty and whether you update on evidence. Structure: what you believed and why it was reasonable, the evidence that changed it, and what you now do differently. Example: "I used to treat RL on verifiable rewards as the main lever for reasoning, because pass@1 kept rising. Then I compared pass@k curves: in a toy run pass@1 rose from 0.34 to 0.47 while pass@64 fell from 0.91 to 0.76, because a rare strategy was squeezed out (Chapter 19.6). Now I track pass@k and entropy in every RL run, and I treat the base model's coverage as a ceiling unless exploration is kept alive." Weak answers name a trivial change or none. Follow-up: "What would change your mind back?" Name the evidence, such as RL solving problems the base model never solves at large k.

**7. Your lead asks you to test method X so it can go into the next pretraining run, which is frozen in three weeks. How do you plan it?** Start from the decision and the bar, not the experiment: agree what result adopts X (for example a compute multiplier of at least 1.1× at the largest scale tested, holding with scale) and the default if unclear (keep the recipe). Then size the options: with 16 H100s, about 8,000 GPU-hours, a 150M/400M/1B ladder with per-arm learning-rate sweeps and 3 seeds costs about 750 GPU-hours, and one 3B run per arm another 1,500. Recommend the ladder first and the 3B pair only if 1B clears the bar, which is the kill criterion. Reject a gain that shrinks with scale even if it is positive everywhere (39.10). Follow-up: "Why tune X separately?" An untuned new method against a tuned baseline is the commonest false negative.

**8. A large pretraining run's loss spikes on day 12 of 30 and settles 0.03 above trend. What do you do?** Price both sides first: on 2,048 GPUs a 4-hour rollback costs about 8,200 GPU-hours (about $16,000 at $2 an hour), while the remaining 18 days are about 885,000 GPU-hours. Then check the hypotheses in order of cost: the data at the spike step, per-rank faults, gradient-norm and logit history, recent changes, all minutes from logs. Roll back and skip the batch or cordon the node if one of those explains it; apply a stability fix before continuing if logits were growing; otherwise continue with a pre-set threshold for rolling back. Tell the run owner within the hour, with a recommendation (39.11). Follow-up: "When would you just push through?" When the gap closes within a day and no cause points to lasting damage.

## Practice

**1. Trade-offs.** For three past projects, write one paragraph each: the trade-off, the alternative you rejected, and the signal that would have changed your mind.

<details><summary>Hint and expected result</summary>

Each paragraph should fit the pattern of 39.1: two options, the axis each wins on, the constraint that decided it, and a measurable signal ("if latency above 200 ms…", "if the gain shrinks below 1.1× at 1B"). If you cannot name the signal, the decision was not really a trade-off; pick another project.

</details>

**2. Lead with the recommendation.** Rewrite one past analysis summary so the recommendation is the first sentence.

<details><summary>Hint and expected result</summary>

Test it on a colleague: after reading only the first two sentences, they should be able to say what you recommend, how confident you are (a range), and what would change it. If they cannot, move the caveats after the evidence.

</details>

**3. One-minute story.** Practise your one-minute story aloud, recorded, until it sounds natural.

<details><summary>Hint and expected result</summary>

Target 50–70 seconds with each of the four parts in 39.7 present. Listen for one number per project mentioned. Re-record until two consecutive takes are within 10 seconds of each other without notes.

</details>

**4. Mock design rounds.** In every mock system-design round, say out loud: the user, the metric, the baseline, the first version, the risk.

<details><summary>Hint and expected result</summary>

Have the interviewer note the time at which each of the five appears. Target: all five within the first 3 minutes. For a research design question, substitute "the decision the result informs" for the user and "the compute budget" for the first version.

</details>

**5. Place yourself on the rubric.** Score yourself mid, senior or staff on each of the five axes in 39.12, ask a colleague to score you independently, and compare.

<details><summary>Hint and expected result</summary>

Expect one or two axes where your colleague scores you lower than you do; those are the ones to work on, and the stories you tell for them are the ones to strengthen first (Chapter 38.10). Write one sentence per axis on what evidence would move you up a column.

</details>

**6. Research exercise: scope and run a decision-shaped ablation.** Pick a small recipe change (a learning-rate schedule, a data filter, an optimiser setting) and plan it as in 39.10: write the decision, the bar, the default and the kill criterion before running anything. Then run it on a small model at two sizes with three seeds per arm, and report the result as a recommendation with an interval.

<details><summary>Hint and expected result</summary>

With nanoGPT-scale models (10M and 40M parameters) this fits on one GPU in a day. Expected outcome: a seed standard deviation for the final validation loss, the difference between arms with a 95% interval at each size, and whether the difference grows or shrinks between sizes. The exercise succeeds if your pre-written bar decides the outcome; if you find yourself changing the bar after seeing the results, note it, because that is the habit the exercise is meant to break.

</details>

## Go deeper

- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Martin Zinkevich, Google): 43 rules on shipping ML, starting with "don't be afraid to launch a product without machine learning".
- [StaffEng](https://staffeng.com/) (Will Larson): stories and guides on influence, writing and technical leadership; also his book *Staff Engineer* (2021).
- Tanya Reilly, *The Staff Engineer's Path* (O'Reilly, 2022).
- [Trustworthy Online Controlled Experiments](https://experimentguide.com/) (Kohavi, Tang and Xu, 2020): deciding from experiments.
- [Eugene Yan's writing](https://eugeneyan.com/writing/) on applied ML, product thinking and careers.

## Done when

In a recorded mock, every design answer names the user (or the decision), the metric, the baseline and one trade-off within the first 2 minutes.

Given any choice from your own work, you can state two options, the axis each wins on, and the signal that would change your mind, in under 30 seconds.

You can state a result with a range and a recommendation in two sentences, and explain in one more sentence why "very likely between a and b" is the flat-prior reading of a confidence interval.

Given "test method X for the next run in three weeks" with a GPU budget, you can produce in 10 minutes an options table with GPU-hours, a kill criterion and the default decision, as in 39.10.

Given a loss spike on a large run, you can price rollback against continuing and order four hypotheses by cost of checking, in under 5 minutes.

You have placed yourself on the 39.12 rubric with a colleague's view, and named the one axis you are working on this quarter.

You have three stories of scoping down or saying no, ready in STAR form, and your one-minute "why this company, why now" story is rehearsed.

---

[← 38 · Behavioural interviews](38-behavioural-interviews.md) · [Contents](../README.md#contents) · [40 · Classic models →](40-classic-models.md)
