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
