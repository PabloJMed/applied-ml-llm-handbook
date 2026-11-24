# 38 · Behavioural interviews

Part X · Research craft and interviews. No prerequisites. Chapter 39 covers the senior mindset these stories should show.

## Why it matters

Technical rounds show whether you can do the work; the behavioural round decides whether you will do it well with other people, and at what level. Many hiring decisions at the senior level turn on this round, and it is the easiest one to improve with preparation: candidates who write five strong stories and practise them aloud do markedly better than those who improvise.

The chapter starts with what interviewers actually listen for (38.1) and the STAR structure for telling a story clearly (38.2). It then builds a bank of five story archetypes that cover nearly every question (38.3), maps common questions to them (38.4), and lists the signals of seniority in an answer (38.5). Sections 38.6 to 38.8 cover tricky questions, the questions to ask your interviewers, and the extra stories research roles need. Section 38.9 rewrites one answer from weak to strong. The last four sections are for research and frontier-lab loops, and for the problem most mid-level candidates actually have: how to show senior signals from stories of modest scope (38.10), the mission and safety questions frontier labs ask (38.11), a research story and a failure story written out in full (38.12), and the follow-up probing that tests whether a story is true and the "I" is real (38.13).

A behavioural answer is evidence, not a description of your character. Interviewers want a specific situation, what you did yourself, and a measurable result, told in two or three minutes. Preparing a small set of well-rehearsed stories in advance is the single most effective step, because almost every question can be answered with one of them.

## Key concepts

- **Behavioural question**: "Tell me about a time when…". It assumes past behaviour predicts future behaviour, so it asks for a specific real event, not a hypothetical. [More on the format](https://www.techinterviewhandbook.org/behavioral-interview/).
- **STAR**: Situation, Task, Action, Result, the standard structure for answering. [Overview](https://en.wikipedia.org/wiki/Situation,_task,_action,_result).
- **Competencies or values**: most companies score answers against a published list (ownership, customer focus, bias for action and so on). Amazon's [leadership principles](https://www.amazon.jobs/content/en/our-workplace/leadership-principles) are the best-known example and a useful practice checklist even if you never interview there.
- **Probing (drill-down)**: the follow-up questions that test whether a story is true and the "I" is real; Amazon calls the interviewer who leads them a *bar raiser*. Section 38.13 has sample sequences.
- **Level signals**: what separates mid-level from senior answers: scope, ambiguity handled, influence beyond your team. Will Larson's [StaffEng](https://staffeng.com/) site describes these well.

## 38.1 What the round tests

Behavioural interviewers listen for evidence, in your past behaviour, of:

- **Ownership:** you took responsibility for outcomes, not just tasks, and followed through.
- **Impact:** measurable results that mattered to the business.
- **Judgement:** sensible trade-offs, simple solutions first, knowing when to stop.
- **Collaboration and influence:** working across teams, handling disagreement, persuading without authority.
- **Learning and growth:** what you learned from failure and changed afterwards.
- **Communication:** explaining technical work clearly to non-technical people.
- **Customer and business focus:** connecting models to users and money.

Read the target company's values or competencies page before the interview and map each of your stories to one or two of them.

## 38.2 The STAR structure

![STAR · a 2–3 minute answer](images/38-diagram-1.png)

*STAR · a 2–3 minute answer*

- **Situation:** just enough context for the story to make sense: team, product, problem, why it mattered. Two or three sentences.
- **Task:** your specific responsibility and the goal.
- **Action:** the bulk of the answer: what *you* did, the options you considered, why you chose one, how you dealt with obstacles and people. This is where seniority shows.
- **Result:** concrete outcomes with numbers (quality, cost, time, revenue, adoption), plus what you learned or would do differently.

**Rules that make answers strong:** say "I" for your part and "we" only for the team's; quantify ("cut labelling cost by 60%", not "made it cheaper"); name a trade-off and a decision; keep each story to 2–3 minutes and let the interviewer pull for more; finish with a lesson. Prepare follow-up depth: interviewers often ask "why that approach?", "what would you do differently?", "what did your manager think?".

## 38.3 Build a story bank: five archetypes

Five well-prepared stories cover nearly every behavioural question, because each can be told from several angles. Below are the five archetypes almost every ML candidate needs, each with a fill-in template. The worked examples are a case study from an applied setting (a food-delivery marketplace); each archetype also has a research version, and 38.12 writes out a research story and a failure story in full. Replace the examples with your own real stories; never use an invented one.

### Story 1: A technical decision with business impact

**Shows:** technical ownership, judgement about cost and quality, impact. **Use for:** "your proudest project", "a technical decision you made", "how you balanced cost and quality".

- **S:** [Product, team and the problem.] *Example: merchants needed classifying into ~40* *categories; existing labels were incomplete, which hurt search filters.*
- **T:** [What you owned.] *Example: designing, evaluating and shipping the classifier.*
- **A:** [Baseline, options considered, the choice and why, how you measured, obstacles.] *Example: built a TF-IDF baseline; moved to an LLM with label definitions for rare* *categories; built a 500-item expert test set; moved the weekly job to a batch API and* *trimmed the prompt.*
- **R:** [Quality vs baseline, cost, adoption.] *Example: macro-F1 up from 0.70 to 0.86; cost* *per run down several-fold; three teams use the output.* **Lesson:** [For example: match the serving mode to the latency the business actually needs.]

*Research version:* choosing between two approaches for a training run, such as a model-based quality filter against a cheaper heuristic for a pretraining mixture, by running matched small-scale ablations. The result is the downstream difference at fixed compute, with its interval, and the GPU-hours the decision saved or spent.

### Story 2: Quality and risk in a user-facing system

**Shows:** handling risk, stakeholder management, defining "good". **Use for:** "a time quality mattered more than speed", "working with non-technical stakeholders", "handling ambiguity".

- **S:** [A system where errors hurt users or the business.] *Example: automated* *moderation and summaries of restaurant reviews.*
- **T:** [Your role.]
- **A:** [How you agreed what "good" means with stakeholders; the evaluation you built; how you traded false positives against false negatives; what stayed with humans.]
- **R:** [Volume, precision/recall, time saved, incidents avoided.] **Lesson:** [For example: agree the error trade-off with the business before tuning.]

*Research version:* a capability or safety evaluation for a model release, where a false "safe" is costly. The story is how you agreed the elicitation protocol and the decision threshold with the safety team before seeing results, and what you reported when a result was borderline.

Chapter 28.7 has a full reference design for this kind of system.

### Story 3: Influence across teams

**Shows:** collaboration and influence without authority. **Use for:** "working with another team", "a disagreement and how you resolved it", "getting buy-in".

- **S:** [Two or more teams that needed to agree on something: a shared taxonomy, a metric definition, an API.]
- **T:** [Your part.]
- **A:** [How you aligned definitions; a concrete disagreement and how you settled it with data or a small experiment; how you kept momentum across time zones or priorities.]
- **R:** [What was delivered and adopted.] **Lesson:** [For example: write definitions down early; a small shared evaluation set settles debates faster than meetings.]

*Research version:* getting pretraining, post-training and evaluation teams to agree on one evaluation protocol (prompts, answer extraction, intervals). Section 38.10 tells this story at mid-level scope.

### Story 4: Learning fast in a new area

**Shows:** learning speed, handling change, delivering under uncertainty. **Use for:** "a time you had to learn something quickly", "dealing with change", "unclear requirements".

- **S:** [You moved to a new domain, for example from NLP to recommendations, or took over an unfamiliar system.]
- **T:** [What you had to deliver, by when.]
- **A:** [How you got up to speed: reading, pairing, small experiments; your first concrete contribution; a decision you influenced, such as fixing an evaluation flaw.]
- **R:** [Outcome.] **Lesson:** [For example: start with the evaluation and the logs, not the model.]

*Research version:* moving from supervised fine-tuning to RL post-training, where your first contribution was finding that the verifier accepted malformed answers, before you had trained anything yourself.

### Story 5: A failure, mistake or disagreement

**Shows:** honesty, learning, maturity. Every loop includes a version of "tell me about a failure". Choose a real one with a clear lesson and a change in your behaviour afterwards; avoid fake failures ("I work too hard") and stories that blame others.

- Common good candidates: a model that looked good offline and disappointed online; an estimate that was badly wrong; a disagreement on approach where you were wrong, or right but handled it poorly.
- Structure: what happened, your part in it, what you did to fix it, what you do differently now, and evidence that you did it differently next time.
- The evidence is the part most candidates leave out, and the part interviewers believe. Section 38.12 has a failure story written out in full, ending with the next time the same kind of error happened and was caught.

**Optional sixth story: depth.** A piece of research, a hard debugging session or a technical deep dive you can discuss at length. It is useful for "tell me about a technical challenge" and lets you show depth on the topics the role cares about.

## 38.4 Mapping common questions to stories

| Question | Best story |
|---|---|
| Tell me about yourself | Not a story: a 90-second summary (below) |
| Your proudest project / biggest impact | 1 |
| A difficult technical decision or trade-off | 1 or 2 |
| Working with another team; influencing without authority | 3 |
| A disagreement with a colleague or stakeholder | 3 or 5 |
| Learning something new quickly; handling change | 4 |
| A failure or mistake | 5 |
| Ambiguous requirements | 2 or 4 |
| Explaining something technical to a non-technical audience | 2 |
| Prioritising under pressure; saying no | 1, 2 or 4 (add what you deprioritised) |
| Why this company, why this role | Prepared answer (38.6) |
| Why this lab; the risks of your work; a safety or ethics concern you raised (frontier labs) | Prepared answers (38.11) |
| Impact beyond your own work | A mid-sized story told with senior behaviours (38.10) |

**"Tell me about yourself" (about 90 seconds):** present, past, future. *Now:* your current role and the two or three kinds of problems you work on. *Before:* the path that led here, in two sentences (studies, previous roles, research). *The thread:* what connects it all, for example building applied ML and LLM systems that ship. *Next:* why this role is the natural next step. Write it, time it, and cut it until it fits.

## 38.5 Senior signals interviewers listen for

- **Scope:** problems that span teams or systems, not single tickets.
- **Judgement:** you chose a simple solution first, measured, and added complexity only when needed; you can say what you decided *not* to do.
- **Business framing:** results in user and business terms, not only model metrics.
- **Influence:** you changed what others did through data, prototypes and clear writing.
- **Ownership of outcomes:** you followed the model into production and cared whether it worked there.
- **Raising others:** mentoring, reviews, documentation, sharing tools.
- **Self-awareness:** honest about mistakes and limits; specific about what you learned.

Whatever level you are aiming for, show one or two of these clearly in each story.

## 38.6 Tricky questions

**"Why do you want to leave your current team?"** Never criticise your manager, team or company. Frame it as moving *towards* something: the problems, the scale, the stage of the company, a location change. Keep it short and positive.

**"Why this company?"** Three specific reasons: something about its products or market, something about the team's technical work (read its engineering blog and recent talks), and something about you that fits (a skill you bring, a problem you want to work on). Generic praise sounds unprepared. Frontier labs ask a sharper version of this question, together with questions about safety motivation; 38.11 covers them.

**"What is your biggest weakness?"** A real, job-relevant weakness you are actively fixing, with evidence. Choose a weakness in how you do the job, not in how you interview: "I'm slow in live coding" flags a risk in the very round being scored and says nothing about your work. A better example for a scientist: over-investing in rigour before checking whether the result matters, such as running a full seed sweep and ablation table on an idea the team was unlikely to adopt. What you do about it: before starting, you now write down which decision the result will inform and the smallest experiment that could change that decision; on your last project that cut a planned three-week study to one week, and you ran the full sweep only on the variant that survived. Honest, bounded, improving, and the weakness is the shadow of a strength the role needs.

**"Tell me about a conflict with your manager."** Choose a disagreement about approach, not personality; show that you raised it with data, listened, and committed once a decision was made ("disagree and commit").

**"Where do you see yourself in a few years?"** Growing into someone who owns systems end to end and helps others do the same. Tie it to what the role offers.

**Level and compensation:** know which level you are applying for and why; if asked, say you are confident at that level and keen to grow through impact. Discuss package details with the recruiter at the offer stage, not in technical rounds.

**How levelling works.** Most large companies and labs set the level from the whole loop, not from years of experience: the technical rounds show whether you can do the work, and the behavioural and research-discussion rounds show the scope you operate at. Down-levelling, an offer one level below the target, usually happens when technical scores are strong but stories show execution of well-defined tasks rather than choosing problems, influencing others or handling ambiguity, which is exactly what 38.10 addresses. Some companies match you to a team after the offer, others hire for one team; ask which, because a team-matching process means the hiring manager you meet later also judges fit. Negotiate the level before the money: the level fixes the compensation band and the expectations at your first review, and moving up a level later usually takes a year or more, while the package within a band moves only a little. If you believe the level is wrong, ask what evidence would change it and offer it, for example an extra conversation about your research.

## 38.7 Questions to ask them

Ask two or three; they signal what you care about.

- "What would success look like for this role in the first six months?"
- "Which problems is the team most excited about this year, and where does GenAI fit?"
- "How does the team decide what to build: who brings the problems, and how are projects prioritised?"
- "How do you evaluate LLM features before and after launch?"
- "What does the path to the next level look like here, and what distinguished people who got there quickly?"
- "How does the team share models and data with other teams in the company?"

## 38.8 Stories for research roles

Research-scientist loops add questions that applied loops rarely ask. Prepare a story for each:

| Question | What it tests | A strong story shows |
|---|---|---|
| "Tell me about a result that turned out to be wrong." | Rigour, honesty | How you found the error, what you changed in your process (seeds, baselines, evaluation) |
| "Tell me about a research direction you abandoned." | Judgement | A kill criterion, evidence, the decision and what you salvaged |
| "Tell me about a negative result you are proud of." | Taste, scientific values | Why the question mattered and how the negative result changed what others did |
| "How did you choose your last project?" | Taste | Impact, tractability, your edge; alternatives you considered |
| "Describe a disagreement with a collaborator about a research decision." | Collaboration | Resolved by an experiment or evidence, not authority |
| "Tell me about a time you had to share credit or resolve authorship." | Integrity | Early, explicit agreements; fairness |
| "What's a paper you think is overrated, and why?" | Critical reading | Specific methodological critique, balanced with what it got right |

Research stories still follow STAR, but the "Result" should include what you learned about doing research, not just the outcome. Section 38.12 writes out the first question's answer in full.

## 38.9 A worked answer, before and after

**Question:** "Tell me about a time you disagreed with a teammate."

**Before (a typical first draft):**

> "We had a disagreement about which model to use. My colleague wanted a deep learning model and I thought gradient boosting was better. We discussed it and in the end we went with my approach, and it worked well. I think communication is really important."

What is wrong with it:

- There is no situation or stakes, and no description of the disagreement's substance.
- The resolution is "we discussed it".
- There is no number, and no sign of how the candidate treated the colleague's view.
- The lesson is a platitude.

**After:**

> **S:** "Our team was replacing the delivery-time estimate shown at checkout. Late deliveries against the promise drove about a third of complaints."
>
> **T:** "I owned the model. A senior colleague owned the serving system and preferred a sequence model over recent courier pings. I thought gradient boosting on aggregated features would be as accurate and much easier to run."
>
> **A:** "Instead of arguing, I proposed we agree on the test first: p90 error on two weeks of held-out orders, sliced by city and by peak hours, plus serving latency. I built the boosting model, and helped my colleague get the sequence model onto the same evaluation. The sequence model was 4% better on p90 error overall, but only at peak hours in the two biggest cities, and it doubled serving cost. We agreed on a hybrid: boosting everywhere, with the sequence model's prediction as a feature at peak hours in those cities. The data pipeline it needed was my colleague's idea, and it was better than either of our starting positions."
>
> **R:** "p90 error fell 11% against the old system, late-against-promise complaints fell 18% over the next quarter, and serving cost rose 15% instead of 100%. The lesson I took: when two people disagree about a method, agree on the evaluation first; the argument usually gets smaller or disappears."

Why it works:

- It names a real trade-off: accuracy against cost and complexity.
- It shows the candidate changing the frame from opinions to a shared test.
- It credits the colleague specifically.
- The outcome is quantified.
- The lesson is concrete enough to be reused, so it gives the interviewer something to ask about next.

It also takes under three minutes to say. Practise it aloud with a timer.

## 38.10 Senior signals from mid-sized scope

The problem for most candidates with two or three years of experience is not a lack of stories but a lack of large ones: they have not led cross-team programmes or owned a product. Interviewers know this. At senior level they are not scoring the size of the project; they are scoring the behaviours in 38.5, and whether those behaviours would carry over to a larger one. A mid-sized story told with senior behaviours beats a large story in which you were a passenger.

Senior behaviours show up at small scope in recognisable ways:

- **Something you built that others adopted.** An evaluation harness, a data tool, a training recipe. Adoption by people who did not have to use it is evidence of influence without authority.
- **Killing your own project.** Stopping work you proposed when the evidence said so, and writing up why, shows judgement and compute stewardship (Chapter 39.8).
- **A methodological catch.** Finding that a result, yours or a colleague's, was an artefact of seeds, leakage, contamination or a harness bug, and changing the process so it cannot recur.
- **Turning an ambiguous request into a decision.** Rescoping a vague ask into a two-week experiment with a kill criterion (Chapter 39.3).
- **A decision someone else acted on.** A one-page memo with a recommendation, even if the decision was small.

The difference is in the telling: name the decision, the alternatives you rejected, the people who changed what they did because of you, and a number. Compare two versions of the same story.

**Before (mid-level telling):** "I built an evaluation script for our team's models. It ran the main benchmarks and saved the results to a dashboard. People found it useful."

**After (senior telling):** "Three teams were evaluating the same checkpoints with their own prompts and answer extraction, and their scores for one model on the same maths benchmark spread over 4.1 points, more than most of the gains we were arguing about. Nobody owned evaluation, so I proposed a shared harness: pinned prompts and extraction, a bootstrap interval on every number, and a regression test that re-scores a fixed set of checkpoints whenever the harness changes. I wrote a one-page proposal, got the two other leads to agree the prompt set in one meeting by showing them the 4-point spread, and built it in three weeks. Two months later all three teams used it, about 40 runs a week; at roughly 25 minutes of set-up saved per run, that is about 17 engineer-hours a week. More important, it caught a tokeniser change that would have looked like a 2-point regression. What I'd do differently: involve the post-training team from day one; retrofitting their chat-format evaluations cost a week."

The second version is the same project at the same scope. It is senior because the stakes are in numbers, the influence is visible (two leads persuaded, three teams adopting), there is a decision with a trade-off, and it ends with a lesson. Do not inflate the scope to compensate: interviewers probe (38.13), and being precise about what you owned is itself a senior signal.

## 38.11 Frontier-lab questions: mission and safety

Frontier labs such as Anthropic, OpenAI and Google DeepMind add questions most companies do not ask: why you want to work on frontier AI at this lab, how you think about the risks of the work, and whether you have ever acted on a safety or ethics concern. They are not checking for a correct opinion. They test whether you have thought about the questions yourself, whether your motivation will hold when the work involves trade-offs between speed and caution, and whether you would raise a concern rather than stay quiet.

| Question | What it tests | A strong answer contains |
|---|---|---|
| "Why this lab, and not another?" | Specific knowledge, real motivation | Two or three specifics from the lab's own published work (papers, policies, model documentation, its research agenda), why they matter to you, and how your skills fit a team there. Not generic praise of "safety" or "impact" |
| "How do you think about the risks of the work you'd do here?" | Your own reasoning, calibration | A concrete risk tied to your role (for an evaluation researcher, understating a capability; for a post-training researcher, reward hacking that carries into deployment behaviour), how you would detect it, and where you are uncertain |
| "Tell me about a time you raised a safety, ethics or integrity concern." | Courage, judgement, proportion | A real situation, how you raised it (with evidence, to whom, how fast), what happened, and what changed |
| "What would you not work on?" or "When would you push back on a launch?" | Values with specifics | A line you can name, the reasoning behind it, and how you would escalate inside the organisation |
| "What do you think of our approach to X?" (a scaling policy, a model specification, openness) | Critical engagement | One thing you find convincing and one you would question, with reasons. Labs want people who disagree well |

Prepare by reading the lab's recent safety and policy documents and a few papers from the team you would join. Be honest about your motivation: if you are drawn mainly by the research problems, say so, and say how you think about the responsibility that comes with them. A rehearsed mission statement that does not match the rest of your interview is worse than a candid answer. Avoid both extremes: dismissing risks as science fiction, and catastrophising with no idea of what you would do about it.

**A strong answer to "a time you raised a concern":** "While building a fine-tuning set from support conversations, I found that about 2% of examples still contained phone numbers and addresses, because the redaction step only handled one country's formats. The training run was scheduled for that week. I paused my part of the pipeline, sent my lead and the data-governance owner a short note with ten examples and the count, and proposed a fix: a broader PII detector plus a manual audit of 500 samples. The run moved by four days. The detector is now a required pipeline stage, and the audit found a second leak in another dataset. I'd raise it the same way again: early, with evidence, and with a fix attached."

## 38.12 Two stories in full: a research result that was wrong, and a failure

Section 38.8 lists the research questions; here are two answers written out, with every number traceable (`code/ch38_check.py`). Use them for structure, not content.

### A research story: the gain that was seed noise

**Question:** "Tell me about a result that turned out to be wrong."

> **S:** "I was on a pretraining data team. I had built a filter for low-quality web text and tested it with the team's standard ablation: train a 1B-parameter model on 25B tokens with and without the filter, and compare the average over eight benchmarks."
>
> **T:** "I had to recommend whether the filter went into the data mixture for the next large run, which was being frozen in three weeks."
>
> **A:** "The first comparison showed +1.1 points, and I had already presented it at our weekly meeting. Before the mixture review, a colleague asked how large seed variance was for that setup. I didn't know, so I reran both arms with four more seeds each, about 800 GPU-hours. The seed-to-seed standard deviation of the eight-benchmark average was 0.6 points, so a one-seed-per-arm comparison has a standard error of 0.6 × √2 ≈ 0.85 for the difference: my 1.1 was 1.3 standard errors, well within noise. With five seeds per arm the difference was +0.3 points, 95% interval −0.4 to +1.0. I told the same meeting that week that the earlier number was wrong and why. I recommended against adding the filter for quality reasons; it did remove 8% of tokens with no measurable cost, which mattered for the data budget, and that is how it was eventually used."
>
> **R:** "The mixture decision no longer rested on a noise result. I changed how the team runs ablations: we measured seed variance for the standard set-up once, adopted at least three seeds per arm for any decision, and wrote down that detecting a 0.5-point effect at that scale needs about 23 seeds per arm, so we stopped running ablations for effects that small and moved those questions to lower-noise evaluations. The lesson: measure the noise floor before the effect, and report the interval before the point estimate."

Why it works: the candidate made the mistake, corrected it publicly and quickly, put numbers on how wrong it was, and changed the team's process in a way that is still visible. Every number can be rederived live, which is what the follow-ups in 38.13 test.

### A failure story: the evaluation bug I shipped

**Question:** "Tell me about a failure."

> **S:** "I maintained the maths evaluation for a post-training team."
>
> **T:** "I updated the answer-extraction code to handle a new output format, so the RL team could evaluate their latest checkpoints."
>
> **A:** "I tested the change on the new format but not on older checkpoints. The new extraction was more lenient: it accepted an answer anywhere in the last paragraph. The RL team reported a 3-point gain, and the decision to extend their run was taken partly on that number. A week later I noticed an old baseline scoring higher under my code than in the previous month's report. I re-scored old and new checkpoints under both versions: 2.4 of the 3 points came from my change, and the real gain was 0.6. I told the RL lead and our manager that day, with the table, before starting the fix. The extension continued, since it was already cheap to finish, but the claimed gain was withdrawn from the weekly report."
>
> **R:** "I added a regression test: any change to the harness re-scores a frozen set of 20 checkpoints and fails if a score moves by more than 0.3 points without an approved reason, and every reported number now carries the harness version. **Evidence that I do it differently:** two months later a tokeniser update moved scores on the frozen set by more than a point; the test caught it before any results went out, and it was fixed in a day."

Why it works: a real failure with a consequence, owned without blaming anyone ("I tested only the new format"), quantified (80% of the claimed gain was mine), disclosed fast, fixed at the level of the process, and followed by evidence of the changed behaviour.

## 38.13 Probing: how interviewers test a story

Interviewers rarely accept a story as told. They drill down to check that it is true, that the "I" is real, and how deep your judgement goes. Amazon's bar raisers do this explicitly, and most senior loops do it informally. Each probe has a purpose; knowing it lets you answer the question actually asked. Three sample sequences follow, with what a strong reply contains.

**Sequence 1 · Ownership and depth (on the research story in 38.12)**

1. *"What exactly did you do, and what did your colleague do?"* I designed the filter, ran every ablation and did the analysis; my colleague asked the question about seed variance, and I say so.
2. *"Why didn't you measure seed variance in the first place?"* Honestly: the team's process did not, and I assumed the standard set-up was low-noise. Now I measure the noise floor before the first comparison.
3. *"How did you compute that interval?"* The difference of means over five seeds per arm, standard error 0.6 × √(2/5) ≈ 0.38, times 1.96. If you cannot rederive a number from your own story, the interviewer will doubt the rest.
4. *"Was 800 GPU-hours worth it?"* Yes, against a large run whose data mixture depended on the answer; I would not have spent it for a decision worth less.
5. *"What would you do differently?"* Measure variance before presenting anything, and show intervals from the first slide.

**Sequence 2 · Perspective and conflict (on the disagreement story in 38.9)**

1. *"What would your colleague say about how this went?"* What they would credit and what they would criticise: "that I pushed hard for boosting at first and only came round to the hybrid when the evaluation showed it".
2. *"Was there a moment it got tense? What did you do?"* A specific moment and a specific action, such as moving the discussion to the shared evaluation.
3. *"If the sequence model had won clearly, what would you have done?"* Committed to it and helped ship it; that is the point of agreeing the test first.
4. *"What did your manager think?"* Their view, including any disagreement with yours.

**Sequence 3 · Bar-raiser style: dive deep on a number (on the result in 38.9)**

1. *"You said complaints fell 18%. Compared with what?"* The baseline period, its length, and whether it was seasonally comparable.
2. *"How do you know the model caused it?"* An A/B test or holdout if there was one; if it was a before–after comparison, say so and call the evidence suggestive.
3. *"What else changed that quarter?"* Name the confounders you checked.
4. *"If I called your manager, what number would they give?"* The same one, because it came from a shared report.

**How to handle probes.** Answer the question asked, briefly, then stop. Say "I don't remember the exact figure; it was around X" rather than inventing precision. When a probe exposes a weakness, own it and say what you would do now. The probe that sinks candidates is the one where the "I" collapses into "we" under pressure, so rehearse each story with someone who asks "and what did *you* do?" after every sentence.

## Pitfalls

- **"We" all the way through,** so the interviewer cannot tell what you did.
- **Long situations, short actions.**
- **No numbers** in the result.
- **Fake failures** or failures that blame others.
- **Criticising your current employer, manager or colleagues.**
- **Improvising** instead of practising stories aloud.
- **One story told the same way for every question,** instead of the angle the question asks for.
- **Forgetting the lesson** at the end.

## Interview questions

**1. Tell me about yourself.** Not a story but a 90-second map: now (your role and two or three kinds of problems), before (the path, in two sentences), the thread that connects them, and why this role is the natural next step. Include one number that signals level, such as "an evaluation harness three teams now use", and name at most three projects so the interviewer can choose one to probe. Weak answers recite the CV in order for five minutes. Follow-up you will get: "Tell me more about X", so every project you mention needs a full STAR story behind it.

**2. Tell me about a result that turned out to be wrong.** It tests rigour and honesty. Choose a real case where you found the error, or someone found it in your work; say how wrong it was in numbers, how fast and how openly you corrected it, and which process you changed (seeds, baselines, harness tests). Section 38.12's example: a +1.1-point single-seed gain became +0.3, interval −0.4 to +1.0, once seed noise (σ = 0.6 points) was measured; the standard error of a one-seed comparison was 0.85. Avoid trivial errors and errors that were someone else's. Follow-up: "Why didn't you catch it earlier?" Answer honestly, then say what you check first now.

**3. Tell me about a failure.** A real failure with a consequence, told in five parts: what happened, your part in it, what you did to fix it, what you do differently, and evidence that you did it differently the next time. The evidence is what makes it credible: in 38.12's example, a regression test that later caught a tokeniser change before results went out. Quantify the damage (2.4 of a reported 3 points came from the candidate's own bug) and the speed of disclosure. Avoid fake failures ("I care too much") and stories that blame others. Follow-up: "How did the people affected react?" Say what they said, including criticism.

**4. Why this lab, and not another?** Two or three specifics from the lab's own published work (papers, policies, model documentation, research agenda), why each matters to you, and how your skills fit a team there. Generic praise of "safety" or "scale" sounds unprepared, because every lab hears it. If your motivation is mainly the research problems, say so honestly and say how you think about the responsibility that comes with them. Follow-up: "What do you disagree with in our approach?" Have one considered disagreement with a reason, and one thing you find convincing.

**5. How do you think about the risks of the work you'd do here?** Name a concrete risk tied to your own role, not to AI in general: for an evaluation researcher, understating a dangerous capability because the elicitation was weak; for a post-training researcher, reward hacking that carries into deployment. Say how you would detect it (a stronger elicitation, held-out environments, reading high-reward samples) and where you are uncertain. Trade-off worth naming: caution costs speed, and you should say how you would judge which matters more in a given case. Follow-up: "What would make you raise an alarm?" A specific trigger and the channel you would use.

**6. Tell me about a time you raised a safety, ethics or integrity concern.** It tests courage and proportion: neither silence nor alarm. Structure: what you noticed, the evidence you gathered, whom you told and how fast, the fix you proposed, and what changed. Section 38.11's example: about 2% of a fine-tuning set still contained personal data; the candidate paused the pipeline, sent ten examples and a count to the owners, proposed a fix, and the run moved by four days. If you have no such story, say so and describe how you would act, but most people have a small one (data use, a misleading result, a shortcut in a review). Follow-up: "What if your manager had said ship anyway?" Escalate through the proper channel, with the evidence.

**7. Tell me about a time you disagreed with a more senior colleague.** Choose a disagreement about substance, settled by evidence rather than rank. Show that you understood their view, proposed a test both sides accepted, and committed to the outcome, especially if it went against you. Section 38.9 is a worked example: agreeing the evaluation first turned an argument into a hybrid that cost 15% more to serve instead of 100%. Weak answers end with "and I was right". Follow-up: "What would they say about you?" A fair account, including where they would criticise you.

**8. What is your biggest weakness?** A real weakness in how you do the job, not in how you interview, with what you are doing about it and evidence it is working. For a scientist, a good one is over-investing in rigour before checking whether the result matters; the fix is writing down which decision a result informs before starting, which in 38.6's example cut a three-week study to one. Avoid strengths in disguise and weaknesses that disqualify you for the role. Follow-up: "How would your manager describe it?" The same way, which is why it has to be real.

**9. Tell me about a project you stopped.** It tests judgement and compute stewardship. Give the kill criterion you set (ideally before starting), the evidence that met it, the cost you avoided, and what you salvaged: a dataset, a negative result others used, a tool. Example shape: "After two weeks the method gave +0.2 points at 400M, interval −0.3 to +0.7, against the +1 point we needed to justify a large run, so I stopped and wrote a two-page note; the data loader I built is now the team default." Follow-up: "Was it the right call in hindsight?" Say what you learnt later, even if it complicates the story.

**10. Tell me about a time your impact reached beyond your own work.** The question mid-level candidates find hardest. Use a mid-sized story told with senior behaviours (38.10): something you built that others adopted, a methodological catch that changed a process, or an ambiguous request turned into a decision. Give adoption in numbers (three teams, 40 runs a week), the people you persuaded and how, and the trade-off you chose. Do not inflate the scope; precision about what you owned is itself a senior signal. Follow-up: "What would have happened without you?" A concrete counterfactual: three teams still reporting scores 4 points apart.

## Practice

**1. Write all five stories in full** in a private document, with real numbers. Then cut each to a 2–3 minute spoken version.

<details><summary>Hint and expected result</summary>

A written story runs to about 400–500 words; the spoken version to about 300–400 words, which is two to three minutes at a normal pace. Each should contain at least one "I decided", one rejected alternative and one number in the result. If a story has no number, find the closest measurable proxy (time saved, adoption, error rate) before you rehearse it.

</details>

**2. Record yourself** answering "tell me about yourself" and two story questions; listen for "we", filler words and missing numbers.

<details><summary>Hint and expected result</summary>

Count three things on playback: uses of "we" where you mean "I", seconds spent on the situation, and numbers stated. Target: the situation under 30 seconds, the action at least half the answer, at least two numbers per story, and "we" only for genuinely shared work.

</details>

**3. Angle drill:** tell story 1 three times: as a technical decision, as a cost trade-off, and as a stakeholder story.

<details><summary>Hint and expected result</summary>

The facts stay the same; what changes is which part of the action you expand and which lesson you end on. If you cannot find a stakeholder angle, the story may be too narrow to carry more than one question; pick a richer one.

</details>

**4. Mock round:** ask a friend or colleague to run 30 minutes of questions from the table in 38.4, with follow-ups.

<details><summary>Hint and expected result</summary>

Give the interviewer the three probe sequences in 38.13 and ask them to use at least one per story. You are done when every probe gets an answer of under 30 seconds that keeps the "I" and the numbers intact.

</details>

**5. Values mapping:** read the target company's values page and map each story to one or two values.

<details><summary>Hint and expected result</summary>

Every value should be covered by at least one story, and no story should map to more than three. For a frontier lab, add its mission and safety questions (38.11) and check that at least one story shows you acting on a concern.

</details>

**6. Questions:** write three questions for each interviewer type (hiring manager, peer, skip-level).

<details><summary>Hint and expected result</summary>

Good questions could not be answered from the company website: how the team decides what to train next, what the last surprising evaluation result was, what distinguished people who were promoted quickly. Drop any question whose answer would not change your decision to join.

</details>

**7. Research exercise: measure your own noise floor, then retell the story.** Take a past experiment you report as a win, ideally one cheap enough to rerun. Rerun both arms with at least three, better five, seeds and compute the difference of means with its standard error, σ·√(2/n) for n seeds per arm. Then rewrite the story's result with the interval.

<details><summary>Hint and expected result</summary>

Expected outcome: the interval, not just the point estimate. If the original gain is under about two standard errors of a one-seed comparison (2 × 0.85 ≈ 1.7 points when σ = 0.6, as in 38.12), expect it to shrink or vanish. Either way you gain a story: a confirmed result now carries an interval, and a vanished one becomes your "result that turned out to be wrong" (38.12). Seeds needed per arm to detect an effect δ at 80% power are about 2 × 7.84 × σ²/δ²: 23 for δ = 0.5 and 6 for δ = 1 when σ = 0.6 (`code/ch38_check.py`).

</details>

## Go deeper

- [Tech Interview Handbook: behavioural interviews](https://www.techinterviewhandbook.org/behavioral-interview/): question bank and answer structure.
- [Amazon leadership principles](https://www.amazon.jobs/content/en/our-workplace/leadership-principles): a widely used competency list to practise against.
- [StaffEng](https://staffeng.com/) by Will Larson: what senior and staff behaviour looks like in practice.
- Tanya Reilly, *The Staff Engineer's Path* (O'Reilly, 2022): chapters on influence and project leadership.
- Gayle Laakmann McDowell, *Cracking the Coding Interview*: the behavioural chapter, for structure.
- Chapter 39 of this guide, for the senior mindset your stories should demonstrate.

## Done when

All five stories are written with real numbers and fit in 2–3 minutes spoken: time each one, and none runs over 3 minutes.

You can tell each story from at least two angles. Your 90-second "tell me about yourself" is rehearsed and timed under 100 seconds. Your answers to "why leave", "why this company" and "biggest weakness" are positive, honest and under a minute each.

A peer who hears each story once can tell you your specific contribution and its main number.

You can rederive every number in your research story live, as in Sequence 1 of 38.13, including the interval.

You have one story showing senior behaviour from mid-sized scope (38.10), one failure story ending with evidence of changed behaviour, and, for frontier labs, a specific answer to "why this lab" and to "a time you raised a concern".

You have three questions ready for each interviewer, and you have done at least one full mock behavioural round in which the interviewer used the probe sequences of 38.13.

---

[← 37 · ML depth question bank](37-ml-depth-question-bank.md) · [Contents](../README.md#contents) · [39 · Thinking like a senior →](39-thinking-like-a-senior.md)
