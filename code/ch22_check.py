import hashlib, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel, ValidationError
from typing import Literal

# dynamic few-shot
class ExampleSelector:
    def __init__(self, examples):
        self.examples = examples
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([e["input"] for e in examples])
        self.X = self.vec.transform([e["input"] for e in examples])

    def select(self, query, k=3):
        sims = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        return [self.examples[i] for i in np.argsort(-sims)[:k]]

examples = [
    {"input": "Farmacia Lopez", "label": "pharmacy"},
    {"input": "Pizzeria Da Mario", "label": "restaurant"},
    {"input": "Supermercado Sol", "label": "grocery"},
    {"input": "Parafarmacia Nova", "label": "pharmacy"},
    {"input": "Burger Bros", "label": "restaurant"},
]
sel = ExampleSelector(examples)
print([e["input"] for e in sel.select("Farmacia del Mar", k=2)])

def build_prompt(query, shots):
    lines = ["Classify the merchant into one of: restaurant, grocery, pharmacy, other.", ""]
    for s in shots:
        lines += [f"Merchant: {s['input']}", f"Category: {s['label']}", ""]
    lines += [f"Merchant: {query}", "Category:"]
    return "\n".join(lines)
print(build_prompt("Farmacia del Mar", sel.select("Farmacia del Mar", 2)))

# structured output with validation + retry
class MerchantLabel(BaseModel):
    category: Literal["restaurant", "grocery", "pharmacy", "other"]
    confidence: Literal["high", "medium", "low"]

def call_with_validation(llm, prompt, schema, max_retries=2):
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries + 1):
        raw = llm(messages)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Invalid output: {e.errors()[0]['msg']}. "
                                                     f"Return only JSON matching the schema."}]
    raise ValueError("no valid output after retries")

replies = iter(['{"category": "pharma", "confidence": "high"}', '{"category": "pharmacy", "confidence": "high"}'])
fake_llm = lambda messages: next(replies)
print(call_with_validation(fake_llm, "x", MerchantLabel))

# prompt versioning
class PromptRegistry:
    def __init__(self):
        self.prompts = {}
    def register(self, name, template, model, params):
        body = json.dumps({"template": template, "model": model, "params": params}, sort_keys=True)
        version = hashlib.sha256(body.encode()).hexdigest()[:10]
        self.prompts[(name, version)] = {"template": template, "model": model, "params": params}
        return version
reg = PromptRegistry()
v = reg.register("merchant_classifier", "Classify: {merchant}", "some-model", {"temperature": 0})
print(v)

# cost with caching
def monthly_cost(calls, static_tokens, dynamic_tokens, output_tokens,
                 usd_per_m_in, usd_per_m_out, cache_read_factor=None, cache_hit_rate=0.0):
    if cache_read_factor is None:
        input_cost = calls * (static_tokens + dynamic_tokens) * usd_per_m_in / 1e6
    else:
        static = static_tokens * (cache_hit_rate * cache_read_factor + (1 - cache_hit_rate))
        input_cost = calls * (static + dynamic_tokens) * usd_per_m_in / 1e6
    return input_cost + calls * output_tokens * usd_per_m_out / 1e6

no_cache = monthly_cost(3_000_000, 3000, 500, 150, 3.0, 15.0)
cached = monthly_cost(3_000_000, 3000, 500, 150, 3.0, 15.0, cache_read_factor=0.1, cache_hit_rate=0.9)
print(round(no_cache), round(cached))

# ---- additions ----
import math
from scipy.stats import binomtest

# 22.2 contextual calibration
cf = np.array([0.7, 0.3]); x = np.array([0.6, 0.4])
cal = (x / cf) / (x / cf).sum()
print("contextual calibration:", cal.round(2).tolist())

# 22.3 reasoning budget
print("thinking 5k tokens at $15/M: $%.3f per call, x%.0f a 150-token answer, $%.0f per 1M calls, %d s at 100 tok/s"
      % (5000 * 15 / 1e6, 5000 / 150, 5000 * 15, 5000 / 100))

# 22.8 attack success rate intervals
for k in (6, 0):
    ci = binomtest(k, 30).proportion_ci(method="wilson")
    print(f"ASR {k}/30: 95% Wilson [{ci.low:.3f}, {ci.high:.3f}]")

# 22.10 test-set sizes
print("n for 5 points at d=0.10: %.0f; detectable gain on 200 dev: %.1f points; 95%% half-width at 85%% on 300: %.1f; at 80%% on 100: %.1f"
      % ((1.96 + 0.84) ** 2 * 0.10 / 0.05 ** 2, 100 * 2.8 * math.sqrt(0.10 / 200),
         196 * math.sqrt(0.85 * 0.15 / 300), 196 * math.sqrt(0.8 * 0.2 / 100)))
print("eval run cost, 500 tickets: $%.2f" % (500 * (3500 * 3 + 150 * 15) / 1e6))

# 22.11 agent context: 40 turns, 10k static prefix, 2k tokens added per turn
S, add, T = 10_000, 2_000, 40
ctx = [S + add * (t - 1) for t in range(1, T + 1)]
uncached = sum(ctx) * 3 / 1e6
cached = ctx[0] * 3 / 1e6 + sum(ctx[t - 1] * 0.3 / 1e6 + add * 3 / 1e6 for t in range(1, T))
print(f"agent: last context {ctx[-1]:,} tokens, total {sum(ctx):,}; ${uncached:.2f} uncached vs ${cached:.2f} cached (x{uncached/cached:.1f})")

# Interview Q7 and Q11
print("long context vs RAG: $%.2f vs $%.3f (x%.0f); cached long context $%.2f" % (200e3 * 3 / 1e6, 5e3 * 3 / 1e6, 200 / 5, 200e3 * 0.3 / 1e6))
print("SE at 88%% on 500 items: %.1f points" % (100 * math.sqrt(0.88 * 0.12 / 500)))
