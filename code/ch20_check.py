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
