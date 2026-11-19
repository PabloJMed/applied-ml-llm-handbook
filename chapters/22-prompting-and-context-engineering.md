# 22 · Prompting and context engineering

Part VI · Inference and prompting. Assumes you have used LLM APIs. Chapter 21 covers decoding and serving, Chapter 33 tools and agents, Chapter 24 evaluation, which every prompt change depends on. Worked design that builds on it: E.3 LLM taxonomy classifier.

## Why it matters

Prompting is the cheapest and fastest lever on an LLM system: a better prompt can move quality more than a model upgrade, in an afternoon instead of a quarter. "Context engineering" is the broader skill: deciding what goes into the model's context window, in what order, in what format, and at what cost, including retrieved documents, tool results, history and examples. In production, most quality, cost and security issues of LLM features trace back to context decisions.

Interviewers for LLM roles probe whether you treat prompts as engineering artefacts (versioned, evaluated, cost-aware, secured) or as magic words. This chapter gives you the techniques and, more importantly, the discipline.

## Key concepts

- **Prompt anatomy**: role and task, context, instructions and constraints, examples, output format. Vendor guides show what works for each model family: [Claude prompt engineering guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview).
- **Zero-shot and few-shot prompting**: asking with instructions only, or with worked examples in the prompt; the ability that made GPT-3 famous. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- **Chain-of-thought**: asking the model to reason step by step before answering improves multi-step problems. [Chain-of-thought paper](https://arxiv.org/abs/2201.11903).
- **Self-consistency**: sample several reasoning paths and take the majority answer. [Self-consistency paper](https://arxiv.org/abs/2203.11171).
- **Structured output**: asking for JSON that matches a schema, enforced by constrained decoding where available (Chapter 21).
- **Context engineering**: choosing what goes into the context window (instructions, retrieved documents, memory, tool results) and in what order; models use the start and end of long contexts best. [Lost in the Middle](https://arxiv.org/abs/2307.03172).
- **Prompt optimisation**: treating prompts as parameters tuned against a metric. [DSPy](https://arxiv.org/abs/2310.03714).
- **Overview of techniques**: Lilian Weng's [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) survey.

## 22.1 Anatomy of a prompt

Every production prompt has the same parts, whether written as one string or as chat messages:

![the context window · stable prefix first, per-request content last](images/15-01.png)

*the context window · stable prefix first, per-request content last*

- **System prompt:** who the model is, what it must and must not do, tone, safety rules. Highest priority, set by you, never by the user.
- **Task instructions:** the specific job, with definitions (what counts as "grocery"?), constraints, and the output format.
- **Examples:** a few input-output pairs showing hard cases and the exact format.
- **Context:** retrieved documents, database records, tool outputs. Mark it clearly as data, not instructions (22.8).
- **History:** earlier turns, trimmed or summarised to fit.
- **User input:** last, so the model reads it with everything else already in place.

**Writing instructions that work:**

- Be specific and positive: "Answer in at most 3 sentences, in the user's language" beats "Don't be too long".
- Define every label and edge case; the model cannot read your mind or your taxonomy wiki.
- Explain *why* a rule exists when it is not obvious; models generalise better from reasons than from bare rules.
- Use clear delimiters (XML-style tags or markdown headers) between instructions, context and input.
- Give the model a way out: "If the documents do not contain the answer, say you don't know" or an `"unsure"` label. Without one it will guess.
- Put stable content first and variable content last: it is clearer, and it makes prompt caching possible (22.7).

## 22.2 Zero-shot and few-shot prompting

**Zero-shot:** instructions only. Works well for tasks the model already understands (summarise, translate, classify into obvious categories) when the instructions and definitions are precise.

**Few-shot:** add a handful of input-output examples. Examples teach format and edge cases better than descriptions do. Rules of thumb:

- Pick examples that cover **hard and boundary cases**, not easy ones the model would get right anyway.
- Keep labels balanced across examples; models pick up the label distribution of the examples.
- Keep the examples' format exactly the format you want back.
- Three to eight examples is usually enough; more costs tokens with fading returns.

**Dynamic few-shot.** Instead of fixed examples, retrieve the most similar labelled examples for each input. This often beats any fixed set, and the example pool becomes a place to store fixes: when the model gets a case wrong, add the corrected case to the pool.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ExampleSelector:
    def __init__(self, examples):
        self.examples = examples
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([e["input"] for e in examples])
        self.X = self.vec.transform([e["input"] for e in examples])

    def select(self, query, k=3):
        sims = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        return [self.examples[i] for i in np.argsort(-sims)[:k]]

def build_prompt(query, shots):
    lines = ["Classify the merchant into one of: restaurant, grocery, pharmacy, other.", ""]
    for s in shots:
        lines += [f"Merchant: {s['input']}", f"Category: {s['label']}", ""]
    lines += [f"Merchant: {query}", "Category:"]
    return "\n".join(lines)
```

Use embeddings instead of TF-IDF for semantic similarity. One trap: nearest neighbours often all share one label ("Farmacia del Mar" retrieves two pharmacies), which can push the model towards that label even when it is wrong. Mixing in a contrasting example from another class keeps the model honest.

## 22.3 Reasoning: chain of thought and reasoning models

**Chain of thought (CoT):** asking the model to reason step by step before answering improved accuracy on multi-step problems (arithmetic, logic, multi-hop questions) in the original studies, and the effect is strongest for larger models.

**Reasoning models** are trained with reinforcement learning to produce long internal reasoning before answering (Chapter 15). With them, you usually do not need to write "think step by step"; instead you control a reasoning budget (effort or thinking-token settings, depending on the provider).

**When reasoning helps:** multi-step logic, maths, planning, applying several rules, tricky classification with interacting conditions. **When it does not:** simple extraction, lookups, short classification with clear labels. There it adds latency and cost for little gain.

**Practical patterns:**

- **Reason, then answer in a fixed field.** Ask for `{"reasoning": ..., "label": ...}` with reasoning first, so the label is conditioned on it. Parse only the label.
- **Self-consistency:** sample several reasoning paths at temperature > 0 and take the majority answer. More accurate, several times the cost; useful for offline labelling, rarely for online traffic.
- **Do not trust the reasoning text as an explanation.** Research on CoT faithfulness shows the stated reasoning does not always reflect what drove the answer. Use it for debugging, not as an audit trail.

## 22.4 Structured output

Production code needs outputs it can parse. Three levels of strictness:

1. **Instructions only** ("answer in JSON with keys..."): works most of the time; breaks at scale.
2. **JSON mode / structured outputs:** the API constrains decoding to valid JSON, often to a given JSON schema. Most major providers support this.
3. **Function or tool calling:** the model fills arguments of a declared function with a schema; the natural choice when the output triggers an action (Chapter 33).

Under the hood, constrained decoding masks tokens that would break the grammar at each step (Chapter 21). Open-source tools such as Outlines, guidance and llama.cpp grammars do the same for self-hosted models.

**Always validate anyway.** A schema guarantees shape, not truth: the label can still be wrong, and an enum can hide a guess. Validate with Pydantic, and on failure retry once with the error message:

```python
from typing import Literal
from pydantic import BaseModel, ValidationError

class MerchantLabel(BaseModel):
    category: Literal["restaurant", "grocery", "pharmacy", "other"]
    confidence: Literal["high", "medium", "low"]

def call_with_validation(llm, prompt, schema, max_retries=2):
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_retries + 1):
        raw = llm(messages)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Invalid output: {e.errors()[0]['msg']}. "
                                                     "Return only JSON matching the schema."}]
    raise ValueError("no valid output after retries")
```

Schema design tips: use enums for closed sets; make fields nullable when information may be absent (and say so in the instructions, or the model will invent values); put a free-text `reasoning` field before the decision fields if you want reasoning; keep nesting shallow; name fields clearly, since field names act as instructions.

## 22.5 Templates, versioning and testing

A prompt in production is code: it has inputs, a behaviour, and regressions. Treat it that way.

- **Templates in files, not strings scattered through the codebase,** with named variables.
- **A version for the whole configuration:** the template, the model, and the decoding parameters together. Changing the model with the same prompt is a new version.
- **An evaluation set run on every change** (Chapter 24): the prompt is accepted only if it beats or matches the current version on the set, with no regression on critical slices.
- **Log the version with every output,** so any production output can be traced to the exact prompt that made it.

```python
import hashlib, json

class PromptRegistry:
    def __init__(self):
        self.prompts = {}

    def register(self, name, template, model, params):
        body = json.dumps({"template": template, "model": model, "params": params}, sort_keys=True)
        version = hashlib.sha256(body.encode()).hexdigest()[:10]     # content-addressed version
        self.prompts[(name, version)] = {"template": template, "model": model, "params": params}
        return version
```

In a team, use a prompt-management or LLM-observability tool (Langfuse, LangSmith, Braintrust and similar) or simply prompts in git plus an evaluation script in CI. **Iterate with error analysis, not intuition:** run the set, read the failures, group them into categories, fix the biggest category, re-run. This loop is the core of applied LLM work (Chapter 24).

**Automatic prompt optimisation.** Tools such as DSPy (Chapter 32) treat prompts and few-shot examples as parameters and optimise them against a metric on your evaluation set. Useful once you have a good set and a reliable metric; useless without them.

## 22.6 Long context and its failure modes

Context windows now reach hundreds of thousands of tokens or more, but using them well is not automatic.

- **Lost in the middle.** Liu et al. (2023) found that models used information at the start and end of a long context better than information in the middle. Newer models are better, but position effects remain: put the most important material near the start or end, and the question after the documents.
- **Distraction.** Irrelevant passages lower accuracy, especially ones that look relevant. More context is not free even when it fits.
- **Degradation with length.** Quality on harder tasks (reasoning across many facts, not just finding one) drops as the context grows, well before the advertised limit. Test at the lengths you will actually use; "needle in a haystack" tests only measure simple retrieval.
- **Cost and latency** grow with every input token, on every call.

**Long context or RAG?** Long context is simpler and better when the whole corpus is small, needed together, and queried rarely (one contract, one codebase module). RAG is better when the corpus is large, changes often, is queried constantly, or needs access control per document. Often both: retrieve the right 20 documents, then use long context to reason over all of them.

**Managing conversation history:** keep the last N turns verbatim, summarise older turns, store durable facts separately (memory, Chapter 33), and drop large tool outputs once they have been used.

## 22.7 Prompt caching and cost

Providers cache the processed state of a prompt **prefix**. When the next call starts with the same prefix, the cached part is billed at a fraction of the normal input price and processed faster. Exact discounts, minimum lengths and cache lifetimes differ by provider and change often; some charge extra to write the cache. Check current pricing.

To benefit, the prefix must be **byte-identical**: stable content first (the diagram in 22.1), no timestamps or user names in the system prompt, examples in a fixed order, tool definitions unchanged.

```python
def monthly_cost(calls, static_tokens, dynamic_tokens, output_tokens,
                 usd_per_m_in, usd_per_m_out, cache_read_factor=None, cache_hit_rate=0.0):
    if cache_read_factor is None:
        input_cost = calls * (static_tokens + dynamic_tokens) * usd_per_m_in / 1e6
    else:
        static = static_tokens * (cache_hit_rate * cache_read_factor + (1 - cache_hit_rate))
        input_cost = calls * (static + dynamic_tokens) * usd_per_m_in / 1e6
    return input_cost + calls * output_tokens * usd_per_m_out / 1e6

# 3M calls/month, 3,000 static + 500 dynamic input tokens, 150 output tokens, $3 / $15 per million
monthly_cost(3_000_000, 3000, 500, 150, 3.0, 15.0)
# ~$38,250
monthly_cost(3_000_000, 3000, 500, 150, 3.0, 15.0, cache_read_factor=0.1, cache_hit_rate=0.9)   # ~$16,380
```

The prices here are illustrative, and the sketch ignores any cache-write premium. The lesson holds: when a long fixed prefix dominates the input, caching can cut input cost by more than half. Combined with batch APIs for offline work (Chapter 21), and with a smaller model for easy cases, this is how LLM features become affordable.

![The monthly bill from the example above, split by where the tokens go](images/22-fig-1.png)

*The monthly bill from the example above, split by where the tokens go. Caching 90% of the static prefix removes $21,870. What remains is mostly output, so the next lever is shorter outputs (a label instead of prose, a tighter schema) rather than a shorter prompt.*

## 22.8 Prompt injection and defences

**Prompt injection** is text that the model treats as instructions when you meant it as data. **Direct injection:** the user types "ignore your instructions and...". **Indirect injection:** the instructions hide in content the system reads: a web page, an email, a review, a PDF, a tool result. Indirect injection is the bigger risk for RAG and agents, because the attacker never talks to your system directly.

There is no complete fix today; models cannot reliably separate instructions from data. Defence is layered and assumes the model can be fooled:

| Layer | What it does |
|---|---|
| Least privilege | The model's tools can only do what the task needs; read-only where possible; scoped credentials per user |
| Human confirmation | Irreversible or costly actions (refunds, emails, deletions) need explicit approval outside the model |
| Deterministic checks | Policy limits enforced in code, not in the prompt (worked design E.10) |
| Separation of untrusted content | Mark retrieved content clearly as data; consider a "quarantined" model that reads untrusted text but has no tools, passing only structured results to the privileged model |
| Input and output filtering | Classifiers for known injection patterns; checks that outputs do not leak system prompts, secrets or other users' data |
| Red-teaming and monitoring | An injection test set run on every release; alerts on unusual tool use |

The interview-ready principle: **never let the model be the only thing standing between untrusted text and a consequential action.** Chapter 28 covers jailbreaks and wider safety.

## 22.9 What research says about in-context learning

- **Why it works.** In-context learning emerges with scale and is linked to induction-style circuits (Chapter 26); theoretical work shows transformers can implement simple learning algorithms (such as a gradient-descent step on a linear model) in their forward pass. In practice, demonstrations mainly communicate the format, label space and input distribution: experiments that randomise the labels of few-shot examples often lose surprisingly little accuracy on classification tasks for smaller models, while larger models follow the given labels more closely.
- **Many-shot ICL.** With long contexts, hundreds or thousands of examples can approach fine-tuning performance on some tasks, at higher inference cost; prompt caching makes it practical. The same mechanism powers many-shot jailbreaking (Chapter 27).
- **Prompt sensitivity is a measurement problem.** Small changes in wording, example order or formatting can move accuracy by several points, more for smaller models. When comparing models or prompts, evaluate several templates and report the spread (Chapter 23), and prefer evaluation designs that are robust to format (for example likelihood scoring for base models).
- **Automatic prompt optimisation** (DSPy-style search, or LLM-proposed rewrites scored on a dev set) can beat hand-written prompts, with the usual risk of over-fitting the dev set: keep a held-out test set.

## 22.10 A worked prompt: support-ticket triage

The task is to route incoming support tickets to one of eight queues and extract the order id, returned as JSON. Here is a development loop that would hold up in review.

1. **Build the set first.**
   - Label 300 recent tickets, stratified by queue and language, with 20% of them hard cases (several issues in one ticket, no order id, sarcasm).
   - Freeze 100 of them as a test set and iterate on the other 200.
   - Is 300 enough? To see a 5-point improvement in a paired comparison where versions disagree on about 10% of tickets takes about 2.8² × 0.10 / 0.05² ≈ 310 tickets (Chapter 23). A 2-point change is invisible at this size, so do not chase one.
2. **Write version 1 in the cacheable order.**
   - The stable part comes first: role, queue definitions with one-line boundaries between similar queues, the output schema, and six examples covering the hard cases.
   - The ticket comes last, delimited as data.
   - Use structured output (22.4) so every response parses.
3. **Run it and read the failures.**
   - Group the errors on the dev set. Typical groups: confusing two neighbouring queues, picking the first issue in a multi-issue ticket, and inventing an order id when none is present.
   - Count each group and fix the largest. A boundary sentence between the two queues often helps. So do a rule ("if several issues, route by the one that blocks the customer") and an explicit null ("order_id: null when absent; never guess").
4. **Re-run with paired comparison.**
   - Compare versions on the same tickets. Count the tickets that flipped from wrong to right and from right to wrong, not only the totals.
   - If outputs are sampled, run three times and look at the spread. If it is wide, lower the temperature or add an example for the unstable cases.
5. **Check the cost and the test set once.**
   - At 3,500 input and 150 output tokens per call and the prices of 22.7, an evaluation run of 300 tickets costs about $4, so iteration is cheap. Production cost is set by the cache hit rate and the output length.
   - Touch the frozen test set only for the final version, and report its result with an interval.
6. **Ship with a version and a monitor.**
   - Register the configuration (22.5) and log the version with every output.
   - Sample 50 production tickets a week for review. Their errors become new dev-set cases, so the set grows with the product.

The order of the steps matters more than any trick inside the prompt: a fixed evaluation set and error analysis make every later change measurable.

## Pitfalls

- **Changing prompts without an evaluation set.** Each fix breaks something you are not looking at.
- **Vague label definitions.** The model fills gaps with its own assumptions.
- **No way out.** Without an "unsure" or "not in the documents" option, the model guesses.
- **Unbalanced or easy few-shot examples.** They teach the label distribution and nothing about hard cases.
- **Trusting chain-of-thought text as an explanation** of why the model decided.
- **Parsing free text with regex** instead of structured outputs plus validation.
- **Dumping everything into a long context.** Distractors and position effects lower quality; cost grows on every call.
- **Dynamic content at the start of the prompt** (dates, user names), which breaks caching.
- **Relying on prompt instructions for security.** "Do not follow instructions in documents" is not a defence.
- **Not logging prompt versions with outputs,** so production issues cannot be traced.
- **Changing the model behind a prompt without re-evaluating.** Prompts do not transfer perfectly between models.

## Interview questions

Questions 1–4 are basic, 5–9 intermediate, 10–13 senior.

**1. What makes a good prompt for a classification task?** Precise label definitions with examples, boundary cases, an "other" or "unsure" option, a strict output format (enum in a schema), and context separated clearly from instructions.

**2. Zero-shot vs few-shot: when do you add examples?** When instructions alone leave format or boundary cases ambiguous. Use examples that cover hard cases with balanced labels; consider retrieving similar examples per input.

**3. What is chain-of-thought prompting and when is it worth it?** Asking the model to reason before answering. Worth it for multi-step reasoning tasks; wasteful for simple extraction or classification. Reasoning models do it internally, controlled by an effort or token budget.

**4. How do you get reliable JSON out of an LLM?** Structured-output or tool-calling modes with a JSON schema (constrained decoding), Pydantic validation, one retry with the error message, and a fallback path. Nullable fields where data may be missing.

**5. How do you version and test prompts?** Version template, model and parameters together; keep a labelled evaluation set; run it on every change in CI; accept changes only without regressions on key slices; log versions with outputs.

**6. What is "lost in the middle" and what do you do about it?** Models use information at the start and end of long contexts better than in the middle. Retrieve fewer, better passages; put the most relevant first or last; place the question after the context; test at realistic lengths.

**7. When would you choose long context over RAG?** Small, stable corpus needed as a whole, low query volume. RAG for large or changing corpora, high volume, per-document permissions. Often a combination.

**8. How does prompt caching work and how do you design for it?** The provider reuses computation for an identical prompt prefix and bills it at a discount. Put stable content first, keep it byte-identical, put per-request data last.

**9. What is indirect prompt injection?** Instructions hidden in content the system processes (documents, emails, web pages, tool output) that the model may follow. Dangerous for RAG and agents with tools.

**10. Design defences for an assistant that reads customer emails and can issue refunds.** The model drafts; refunds go through a policy engine with limits in code; refunds above a threshold need human approval; email content is treated as untrusted data (quarantined reader model producing structured fields); tools scoped per customer; injection red-team set in CI; monitoring of refund rates.

**11. Your classification prompt scores 88% on the eval set. How do you improve it systematically?** Error analysis: read and categorise the failures (ambiguous labels, missing context, format errors, genuinely hard cases); fix the largest category first (often definitions or examples); re-run; check slices for regressions. Consider dynamic few-shot, a stronger model only for low-confidence cases, or fine-tuning if the error pattern is consistent.

**12. A new model version is released. How do you decide whether to switch?** Run the full evaluation set with the current prompt and with light prompt adjustments; compare quality per slice, latency and cost; check structured-output reliability and safety tests; shadow-test on production traffic; switch with a flag and monitor.

**13. How would you cut the cost of an LLM feature by 5× without losing quality?** Measure where tokens go; shorten and cache the stable prefix; trim retrieved context; route easy cases to a smaller model (cascade); batch offline work; cache repeated requests; distil high-volume tasks into a small model. Verify each step on the evaluation set.

## Practice

1. **Prompt iteration loop.** Take 200 labelled texts from any classification dataset. Write a zero-shot prompt, measure accuracy, do error analysis, and improve it three times. Keep a table of version, change, accuracy and cost.
2. **Dynamic few-shot.** Compare fixed few-shot with dynamic few-shot (embedding retrieval) on the same set.
3. **Structured outputs.** Implement `call_with_validation` against a real API with a schema, and measure the failure rate with and without structured-output mode on 500 calls.
4. **Long context.** Build a test with 50 documents where the answer sits at different positions; measure accuracy by position and context length for one model.
5. **Caching.** Restructure a prompt so its static prefix is cacheable and measure latency and cost before and after with your provider's usage reports.
6. **Injection red-team.** Write 30 injection attempts hidden in documents for a small RAG demo; measure how many succeed with and without a quarantined-reader design.

## Read next

- *AI Engineering* (Chip Huyen): the chapter on prompt engineering, including defensive prompting.
- Your LLM provider's prompt engineering guide (Anthropic and OpenAI both publish detailed ones) and their docs on structured outputs and prompt caching.
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022); Wang et al., *Self-Consistency Improves Chain of Thought Reasoning* (2022).
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023).
- Greshake et al., *Not What You've Signed Up For: Compromising Real-World LLM-* *Integrated Applications with Indirect Prompt Injection* (2023).
- Simon Willison's writing on prompt injection and the dual-LLM pattern (simonwillison.net).
- Hamel Husain's posts on evaluation-driven prompt iteration.

**Links:** [Claude prompt engineering guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) · [Lilian Weng: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) · [Chain-of-thought](https://arxiv.org/abs/2201.11903) · [Eugene Yan: Patterns for LLM systems](https://eugeneyan.com/writing/llm-patterns/). Other titles above can be found by searching their title.

## Done when

You can name the parts of a production prompt and why stable content goes first. You can design few-shot examples, including dynamic selection and its label-bias trap.

You can say when reasoning helps and how to control it with reasoning models. You can implement structured output with validation and retry. You can describe a prompt versioning and evaluation workflow.

You can explain long-context failure modes and choose between long context and RAG. You can do prompt-caching cost arithmetic. You can explain direct and indirect injection and design layered defences.

---

[← 21 · Inference, decoding and serving](21-inference-decoding-and-serving.md) · [Contents](../README.md#contents) · [23 · Evaluating models →](23-evaluating-models.md)
