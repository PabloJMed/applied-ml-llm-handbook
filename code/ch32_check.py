from typing import TypedDict, Callable
import json
class State(TypedDict, total=False):
    query: str; intent: str; docs: list; answer: str; attempts: int

def classify(s: State) -> State:
    return {**s, "intent": "refund" if "refund" in s["query"].lower() else "faq"}
def retrieve(s: State) -> State:
    kb = {"refund": ["Refunds go to the original payment method within 5-7 days."], "faq": ["Delivery fees depend on distance."]}
    return {**s, "docs": kb[s["intent"]]}
def generate(s: State) -> State:
    return {**s, "answer": f"[{s['intent']}] " + s["docs"][0], "attempts": s.get("attempts", 0) + 1}
def check(s: State) -> str:
    return "end" if s["answer"] and s["attempts"] >= 1 else "generate"

class Graph:
    def __init__(self): self.nodes, self.edges, self.cond = {}, {}, {}
    def add_node(self, name, fn): self.nodes[name] = fn
    def add_edge(self, a, b): self.edges[a] = b
    def add_conditional(self, a, router: Callable[[State], str]): self.cond[a] = router
    def run(self, start, state, max_steps=20):
        node, trace = start, []
        for _ in range(max_steps):
            state = self.nodes[node](state); trace.append(node)
            nxt = self.cond[node](state) if node in self.cond else self.edges.get(node, "end")
            if nxt == "end": return state, trace
            node = nxt
        raise RuntimeError("step limit hit")
g = Graph()
g.add_node("classify", classify); g.add_node("retrieve", retrieve); g.add_node("generate", generate)
g.add_edge("classify", "retrieve"); g.add_edge("retrieve", "generate"); g.add_conditional("generate", check)
out, trace = g.run("classify", {"query": "I want a refund for order 123"})
print(trace, out["answer"])

# 32.9 the same RAG step in plain Python: retries, cache, validation, trace
import hashlib, json, random, time
from pydantic import BaseModel, ValidationError

class Answer(BaseModel):
    answer: str
    citations: list[str]

class TransientError(Exception):
    pass

PROMPT_VERSION = "rag-answer-v3"
PROMPT_V3 = ("Answer from the passages only. Return JSON with 'answer' and 'citations' (passage ids).\n"
             "If the passages do not answer the question, say so.\n\nPassages:\n{passages}\n\nQuestion: {question}")

def call_with_retries(llm, prompt, attempts=3, base_delay=0.5):
    for i in range(attempts):
        try:
            return llm(prompt)
        except TransientError:                       # rate limits, timeouts: retry with jittered backoff
            if i == attempts - 1:
                raise
            time.sleep(base_delay * 2 ** i * (1 + random.random()))

def answer(question, retrieve, llm, cache, trace, model="small-model", temperature=0.0, k=5):
    passages = retrieve(question, k=k)
    trace.append({"step": "retrieve", "ids": [p["id"] for p in passages]})
    prompt = PROMPT_V3.format(passages="\n".join(f"[{p['id']}] {p['text']}" for p in passages), question=question)
    key = hashlib.sha256(json.dumps([model, PROMPT_VERSION, temperature, prompt]).encode()).hexdigest()
    raw = cache.get(key) if temperature == 0 else None   # never serve a sampled output from cache
    cached = raw is not None
    if not cached:
        t0 = time.time(); raw = call_with_retries(llm, prompt)
        trace.append({"step": "llm", "cached": False, "seconds": round(time.time() - t0, 3)})
    else:
        trace.append({"step": "llm", "cached": True})
    try:
        out = Answer.model_validate_json(raw)
    except ValidationError:
        trace.append({"step": "parse", "ok": False}); return None
    unknown = set(out.citations) - {p["id"] for p in passages}
    trace.append({"step": "check", "unknown_citations": sorted(unknown)})
    if unknown:
        return None
    if not cached and temperature == 0:
        cache[key] = raw                                  # cache only outputs that passed every check
    return out

docs = [{"id": "p1", "text": "Refunds go to the original payment method within 5 days."},
        {"id": "p2", "text": "Vouchers expire after 30 days."}]
calls = {"n": 0}
def fake_llm(prompt):
    calls["n"] += 1
    if calls["n"] == 1:
        raise TransientError("429")
    return json.dumps({"answer": "Within 5 days, to the original payment method.", "citations": ["p1"]})
cache, trace = {}, []
r1 = answer("When do I get my refund?", lambda q, k: docs[:k], fake_llm, cache, trace)
r2 = answer("When do I get my refund?", lambda q, k: docs[:k], fake_llm, cache, trace)
print("32.9:", r1.citations, "llm calls", calls["n"], [e["step"] + ("(cached)" if e.get("cached") else "") for e in trace])

# a response with a hallucinated citation is rejected and NOT cached: the next identical call goes back to the model
bad = {"n": 0}
def bad_llm(prompt):
    bad["n"] += 1
    return json.dumps({"answer": "Vouchers last a year.", "citations": ["p9"]})
cache2, trace2 = {}, []
r3 = answer("How long do vouchers last?", lambda q, k: docs[:k], bad_llm, cache2, trace2)
r4 = answer("How long do vouchers last?", lambda q, k: docs[:k], bad_llm, cache2, trace2)
print("32.9 invalid output:", r3, r4, "model calls", bad["n"], "cache entries", len(cache2))
# sampled outputs (temperature > 0) are never served from the cache
calls["n"] = 1
r5 = answer("When do I get my refund?", lambda q, k: docs[:k], fake_llm, cache, [], temperature=0.7)
print("32.9 temperature 0.7: model called again ->", calls["n"] == 2)
