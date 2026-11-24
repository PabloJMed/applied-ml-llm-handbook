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

# 33.2 argument validation against the schema
JSON_TO_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
def validate(args, schema):
    errs = [f"missing argument {k}" for k in schema["required"] if k not in args]
    for k, v in args.items():
        spec = schema["properties"].get(k)
        if spec is None:
            errs.append(f"unknown argument {k}")
        elif "enum" in spec and v not in spec["enum"]:
            errs.append(f"{k} must be one of {spec['enum']}")
        elif not isinstance(v, JSON_TO_PY[spec["type"]]):
            errs.append(f"{k} should be a {spec['type']}")
    return errs
print("33.2 validation:", validate({"order_id": "A1", "amount": "2.5", "reason": "cold"}, issue_voucher.spec["input_schema"]))

# 33.9 recovery and per-task heterogeneity
p, q = 0.95, 0.5
print("33.9 with recovery: per step %.3f, 10 steps %.3f (vs %.3f)" % (p + (1 - p) * q, (p + (1 - p) * q) ** 10, p ** 10))
print("pass^4 for tasks at 0.9 and 0.3 (mean 0.6): %.3f vs 0.6^4 = %.3f" % (0.5 * 0.9 ** 4 + 0.5 * 0.3 ** 4, 0.6 ** 4))

# 33.9 cache-write premium (1.25x on newly cached tokens) and cache expiry
ctx = 4000 + 600 * (steps - 1)
w = np.where(steps == 1, 1.25 * ctx, 0.1 * (ctx - 600) + 1.25 * 600)
w_exp = w.copy(); w_exp[9] = 1.25 * ctx[9]
print("33.9 billed tokens: reads 10%%, writes free %.0f; writes at 1.25x %.0f; cache expires at step 10 %.0f; expires every step %.0f"
      % (np.where(steps == 1, ctx, 0.1 * (ctx - 600) + 600).sum(), w.sum(), w_exp.sum(), (1.25 * ctx).sum()))

# 33.11 loss mask share, synchronous rollouts, environment throughput
print("33.11 generated share of a 3-turn trajectory: %d / %d = %.2f" % (450, 1200 + 150 + 2000 + 200 + 800 + 100, 450 / 4450))
rng = np.random.default_rng(0)
d = rng.lognormal(np.log(120), 1.0, (20000, 64))
print("33.11 rollout seconds: mean %.0f, slowest of 64 %.0f, utilisation %.2f" % (d.mean(), d.max(1).mean(), d.mean() / d.max(1).mean()))
print("33.11 sandbox-hours per batch %.0f; sandboxes for a 10-minute batch %.0f" % (512 * 30 * 15 / 3600, 512 * 30 * 15 / 600))

# 33.13 context strategies: billed input in token-equivalents (cached reads 0.1x, writes 1.25x)
PREFIX, OUT, TOOL, STUB, STEPS, WINDOW = 6000, 300, 2200, 50, 80, 200_000
def bill(prev, cur):
    common = 0
    for a, b in zip(prev, cur):
        if a != b: break
        common += a[1]
    total = sum(s for _, s in cur)
    return 0.1 * common + 1.25 * (total - common), total
def run_ctx(strategy, threshold=100_000, keep=5):
    ctx_, prev, billed, peak, summaries = [("prefix", PREFIX)], [], 0.0, 0, 0
    window = 10 ** 9 if strategy == "unbounded" else WINDOW
    for t in range(1, STEPS + 1):
        b, tot = bill(prev, ctx_); billed += b; peak = max(peak, tot)
        if tot > window:
            return f"overflows at step {t} ({tot} tokens)"
        prev = list(ctx_)
        ctx_ = ctx_ + [(f"out{t}", OUT), (f"tool{t}", TOOL)]
        tot = sum(s for _, s in ctx_)
        clear = strategy == "clear_every_step" or (strategy == "clear_at_threshold" and tot > threshold)
        if clear:
            ctx_ = [(n, STUB) if n.startswith("tool") and int(n[4:]) <= t - keep else (n, s) for n, s in ctx_]
        if strategy == "compact_at_threshold" and tot > threshold:
            billed += 0.1 * tot; summaries += 1                  # the summarising call rereads the cached history
            ctx_ = [("prefix", PREFIX), (f"summary{t}", 4000)] + ctx_[-6:]
    return f"peak {peak}, billed {billed / 1e6:.2f}M, summaries {summaries}"
for strategy in ("none", "unbounded", "clear_every_step", "clear_at_threshold", "compact_at_threshold"):
    print("33.13", strategy, "->", run_ctx(strategy))
print("33.14 50 steps at 98%%: %.2f" % 0.98 ** 50)
