import inspect, json, time
from typing import get_type_hints, Literal, get_origin, get_args

PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}

def tool(fn):
    """Build a JSON-schema tool spec from a function's signature and docstring."""
    hints = get_type_hints(fn)
    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        t = hints[name]
        if get_origin(t) is Literal:
            props[name] = {"type": "string", "enum": list(get_args(t))}
        else:
            props[name] = {"type": PY_TO_JSON[t]}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    fn.spec = {"name": fn.__name__, "description": inspect.getdoc(fn),
               "input_schema": {"type": "object", "properties": props, "required": required}}
    return fn

ORDERS = {"A1": {"status": "delivered", "items": ["burger", "fries", "cola"], "total": 18.5}}
@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id. Returns status, items and total in euros."""
    return ORDERS.get(order_id, {"error": f"order {order_id} not found"})

@tool
def issue_voucher(order_id: str, amount: float, reason: Literal["missing_item", "late", "quality"]) -> dict:
    """Issue a voucher for an order. Amount in euros, at most 10. Requires a reason."""
    if amount > 10:
        return {"error": "amount above policy cap of 10 euros; escalate instead"}
    return {"ok": True, "voucher": amount}

print(json.dumps(issue_voucher.spec, indent=None)[:200])

def run_agent(llm, tools, user_message, max_steps=8, max_seconds=60):
    registry = {t.__name__: t for t in tools}
    messages = [{"role": "user", "content": user_message}]
    start = time.time()
    for step in range(max_steps):
        if time.time() - start > max_seconds:
            return {"status": "timeout", "messages": messages}
        reply = llm(messages, [t.spec for t in tools])
        messages.append({"role": "assistant", "content": reply})
        if reply.get("tool_call") is None:
            return {"status": "done", "answer": reply["text"], "steps": step + 1, "messages": messages}
        call = reply["tool_call"]
        fn = registry.get(call["name"])
        try:
            result = fn(**call["arguments"]) if fn else {"error": f"unknown tool {call['name']}"}
        except Exception as e:
            result = {"error": str(e)}
        messages.append({"role": "tool", "name": call["name"], "content": json.dumps(result)})
    return {"status": "max_steps", "messages": messages}

script = iter([
    {"tool_call": {"name": "get_order", "arguments": {"order_id": "A1"}}},
    {"tool_call": {"name": "issue_voucher", "arguments": {"order_id": "A1", "amount": 2.5, "reason": "missing_item"}}},
    {"tool_call": None, "text": "Sorry about the missing cola. I've added a 2.50 euro voucher to your account."},
])
fake_llm = lambda messages, tools: next(script)
out = run_agent(fake_llm, [get_order, issue_voucher], "My cola was missing from order A1")
print(out["status"], out["steps"], out["answer"])
print([m["content"] for m in out["messages"] if m["role"] == "tool"])

for p in [0.99, 0.95, 0.9]:
    print(p, [round(p ** n, 2) for n in (5, 10, 20)])

# 33.9 cost of a 20-step agent: 4,000-token prefix, +600 tokens of history per step, 200 output tokens per step, $3/$15 per M
import numpy as np
steps = np.arange(1, 21)
for delta, cache in ((600, False), (600, True), (400, True)):
    ctx = 4000 + delta * (steps - 1)
    billed = ctx if not cache else np.where(steps == 1, ctx, 0.1 * (ctx - delta) + delta)
    print(f"delta {delta}, cache {cache}: input ${billed.sum() * 3 / 1e6:.3f} + output ${20 * 200 * 15 / 1e6:.3f}")
print("latency: 20 x (0.6 s to first token + 200 tokens x 15 ms) = %.0f s before tool time" % (20 * (0.6 + 200 * 0.015)))
