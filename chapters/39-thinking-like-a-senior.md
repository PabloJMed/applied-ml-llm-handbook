# 39 · Thinking like a senior

Part X · Research craft and interviews.

## Why it matters

Technical depth gets you into the room; judgement is what interviewers score for seniority. Across every round (coding, ML depth, system design, behavioural) they listen for the same signals: do you start from the problem rather than the model, do you name trade-offs, do you know when to stop, can you say "I'm not sure" usefully, and do other people get better around you? This chapter turns those signals into habits and phrases you can practise. It is short on purpose; the practice is in using it in every other chapter.

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

Most data scientists rarely manage people, but at mid and senior level you are expected to influence:

- **Write things down.** A one-page design doc or decision memo gets more alignment than meetings. State the problem, options, recommendation, risks, and what you need from whom.
- **Make others successful.** Review code and analyses carefully and kindly; pair with juniors; share reusable tools (an evaluation harness, a prompt-testing script).
- **Build trust before you need it.** Deliver small things reliably; be the person whose numbers hold up.
- **Disagree and commit, in that order.** Make your case with data once, clearly; if the decision goes the other way, commit fully and help measure it.
- **Bring problems with options.** "The labelling budget won't cover all 300 attributes. Three options: … I recommend the second."
- **Work across teams.** Most ML impact at a marketplace depends on product, engineering, operations and commercial teams. Learn their goals and vocabulary.

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

Seniority in research looks different from seniority in product engineering:

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

## Pitfalls

- Treating "senior" as "knows more algorithms".
- Hedging everything so nothing can be wrong; that is not the same as communicating uncertainty.
- Over-scoping in system design rounds: designing the five-year platform instead of the first useful version.
- Talking only about the model in a product question.
- Overclaiming ownership. Say "I" for what you did and "we" for the team, accurately.

## Interview questions

1. Tell me about a time you decided not to use ML.
2. Tell me about a trade-off you made between speed and quality. Would you make it again?
3. Your manager asks for a model to predict which restaurants will churn, by next week. What do you do first?
4. You disagree with a product manager about launching a model whose offline metrics look good but whose A/B result is flat. How do you handle it?
5. How do you explain a 95% confidence interval to a non-technical stakeholder?
6. What's something you changed your mind about recently, technically?

Guidance: (1) and (2) are behavioural; use STAR (Chapter 38) and end with what you learned. (3) Clarify the decision (who acts on the prediction, how), define churn, check data and a baseline (recent order trend), deliver a first version and a plan, not a deep model in a week. (4) Trust the online result over offline metrics; look for reasons (metric mismatch, novelty, segments), propose a longer or targeted test, and accept the decision. (5) "If we repeated this test many times, ranges built this way would contain the true effect 95% of the time; practically, the effect is very likely between a and b."

## Practice

- For three past projects, write one paragraph each: the trade-off, the alternative you rejected, and the signal that would have changed your mind.
- Rewrite one past analysis summary so the recommendation is the first sentence.
- Practise your one-minute story aloud, recorded, until it sounds natural.
- In every mock system-design round, say out loud: the user, the metric, the baseline, the first version, the risk.

## Go deeper

- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Martin Zinkevich, Google): 43 rules on shipping ML, starting with "don't be afraid to launch a product without machine learning".
- [StaffEng](https://staffeng.com/) (Will Larson): stories and guides on influence, writing and technical leadership; also his book *Staff Engineer* (2021).
- Tanya Reilly, *The Staff Engineer's Path* (O'Reilly, 2022).
- [Trustworthy Online Controlled Experiments](https://experimentguide.com/) (Kohavi, Tang and Xu, 2020): deciding from experiments.
- [Eugene Yan's writing](https://eugeneyan.com/writing/) on applied ML, product thinking and careers.

## Done when

You answer any design question by starting from user, metric and baseline.

You habitually name two options and the trade-off between them.

You can state a result with a range and a recommendation in two sentences.

You have three stories of scoping down or saying no, ready in STAR form.

Your one-minute "why this company, why now" story is rehearsed.

---

[← 38 · Behavioural interviews](38-behavioural-interviews.md) · [Contents](../README.md#contents) · [40 · Classic models →](40-classic-models.md)
