import json
import numpy as np

def context_recall(retrieved_ids, relevant_ids):
    return len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else None

def context_precision_at_k(retrieved_ids, relevant_ids, k=None):
    """Average precision over the ranked context: rewards relevant passages placed early."""
    ranked = retrieved_ids[:k] if k else retrieved_ids
    hits, total = 0, 0.0
    for i, pid in enumerate(ranked, start=1):
        if pid in relevant_ids:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0

print(context_recall(["a", "b", "c", "d"], {"b", "x"}), round(context_precision_at_k(["a", "b", "c", "d"], {"b", "d"}), 3))

def faithfulness(answer, context, extract_claims, is_supported):
    claims = extract_claims(answer)
    if not claims:
        return None, []
    verdicts = [(c, is_supported(c, context)) for c in claims]
    return sum(v for _, v in verdicts) / len(claims), [c for c, v in verdicts if not v]

fake_claims = lambda a: [s.strip() for s in a.split(".") if s.strip()]
fake_support = lambda c, ctx: all(w in ctx.lower() for w in c.lower().split() if len(w) > 3)
ctx = "vouchers are worth 20% of the order value, up to 10 euros. requests within 24 hours."
print(faithfulness("Vouchers are worth 20% of the order. Refunds arrive within 3 days.", ctx, fake_claims, fake_support))

def abstention_report(should_answer, did_answer, correct):
    sa, da, c = (np.asarray(a, dtype=bool) for a in (should_answer, did_answer, correct))  # bool masks: ~ and & must be logical
    return {
        "answered_and_correct": np.mean(c[sa & da]) if (sa & da).any() else None,
        "wrongly_abstained": np.mean(~da[sa]) if sa.any() else None,
        "wrongly_answered": np.mean(da[~sa]) if (~sa).any() else None,
    }
print(abstention_report([1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 0, 0, 0]))
print(abstention_report(np.array([1, 1, 1, 0, 0], bool), np.array([1, 1, 0, 1, 0], bool), np.array([1, 0, 0, 0, 0], bool)))

def tool_call_scores(expected, actual):
    """expected/actual: lists of (tool_name, args_dict). Order-aware matching by LCS."""
    def same(a, b):
        return a[0] == b[0] and a[1] == b[1]
    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if same(expected[i-1], actual[j-1]) else max(dp[i-1][j], dp[i][j-1])
    matched = dp[n][m]
    names_ok = sum(1 for a in actual if a[0] in {e[0] for e in expected})
    return {"call_recall": matched / n if n else 1.0, "call_precision": matched / m if m else 1.0,
            "known_tool_rate": names_ok / m if m else 1.0, "extra_calls": m - matched}

expected = [("get_order", {"order_id": "A1"}), ("get_courier_status", {"order_id": "A1"}),
            ("issue_voucher", {"order_id": "A1", "amount": 5.0})]
actual = [("get_order", {"order_id": "A1"}), ("get_order", {"order_id": "A1"}),
          ("issue_voucher", {"order_id": "A1", "amount": 8.0})]
print(tool_call_scores(expected, actual))

def check_end_state(state, expected):
    """Compare final environment state with the expected outcome; ignore fields not specified."""
    return {k: state.get(k) == v for k, v in expected.items()}
print(check_end_state({"voucher_issued": 8.0, "escalated": False, "ticket_status": "resolved"},
                      {"voucher_issued": 5.0, "ticket_status": "resolved"}))

# 25.7 pass^k with heterogeneous tasks: per-task success p ~ Beta(3, 1), mean 0.75
import numpy as np
from math import comb
rng = np.random.default_rng(0); T, n = 120, 8
p = rng.beta(3, 1, T); c = rng.binomial(n, p)
est = lambda c, k: np.mean([comb(int(ci), k) / comb(n, k) for ci in c])
for k in (1, 4, 8):
    boot = [est(c[rng.integers(0, T, T)], k) for _ in range(2000)]
    print(f"pass^{k}: {est(c, k):.3f} [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]; "
          f"Beta(3,1) theory {3 / (3 + k):.3f}; if every task had p = 0.75: {0.75 ** k:.3f}")
print("tasks never solved:", int((c == 0).sum()), " always solved:", int((c == n).sum()))

# 25.7 why heterogeneity slows the fall of pass^k: E[p^k] >= (E p)^k (Jensen); for p ~ Beta(a, 1), E[p^k] = a / (a + k)
for k in (1, 4, 8):
    print(f"Beta(3,1): E[p^{k}] = 3/(3+{k}) = {3 / (3 + k):.3f} vs (E p)^{k} = {0.75 ** k:.3f}")
def pass_at_k(c, n, k):  # unbiased: 1 - C(n-c, k) / C(n, k)
    return 1 - comb(n - c, k) / comb(n, k)
def pass_hat_k(c, n, k):  # unbiased: C(c, k) / C(n, k)
    return comb(c, k) / comb(n, k)
print("task with 5 of 8 runs successful: pass@2 %.3f, pass^2 %.3f" % (pass_at_k(5, 8, 2), pass_hat_k(5, 8, 2)))
print("mix average: pass@4 %.3f (theory 1 - E[(1-p)^4] = %.3f), pass^4 %.3f" % (
    np.mean([pass_at_k(int(ci), n, 4) for ci in c]), 1 - 6 / (5 * 6 * 7), np.mean([pass_hat_k(int(ci), n, 4) for ci in c])))

# 25.7 cost-aware comparison. Beta(3,1): E[(1-p)^k] = 6 / ((k+1)(k+2)(k+3))
E1mp = lambda k: 6 / ((k + 1) * (k + 2) * (k + 3))
cost_a, cost_b, p_b = 0.05, 0.40, 4.5 / 5.5           # agent B: per-task p ~ Beta(4.5, 1)
retry_success = 1 - E1mp(3); retry_runs = 1 + E1mp(1) + E1mp(2)
print(f"agent A: success 0.750, ${cost_a:.3f}/task, ${cost_a / 0.75:.3f}/success")
print(f"agent B: success {p_b:.3f}, ${cost_b:.3f}/task, ${cost_b / p_b:.3f}/success")
print(f"A with verifier and up to 3 tries: success {retry_success:.3f} (if every task had p=0.75: {1 - 0.25 ** 3:.3f}), "
      f"expected runs {retry_runs:.2f}, ${cost_a * retry_runs:.4f}/task, ${cost_a * retry_runs / retry_success:.4f}/success")
print(f"evaluation cost: 120 tasks x 8 runs x 12 calls = {120 * 8 * 12} calls")

# 25.8 full state diff catches side effects that a field-by-field check ignores
def state_diff(before, after):
    keys = before.keys() | after.keys()
    return {k: (before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)}
def check_outcome(before, after, expected_changes):
    diff = state_diff(before, after)
    missing = {k: v for k, v in expected_changes.items() if after.get(k) != v}
    unexpected = {k: d for k, d in diff.items() if k not in expected_changes}
    return {"success": not missing and not unexpected, "missing": missing, "unexpected": unexpected}
before = {"order/A1/voucher": 0.0, "ticket/T9/status": "open", "customer/C4/tier": "standard", "order/B7/voucher": 0.0}
after = {"order/A1/voucher": 5.0, "ticket/T9/status": "resolved", "customer/C4/tier": "gold", "order/B7/voucher": 0.0}
expected = {"order/A1/voucher": 5.0, "ticket/T9/status": "resolved"}
print("field check:", check_end_state(after, expected))
print("state diff check:", check_outcome(before, after, expected))

# 25.8 auditing successful trajectories for exploits
import math
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c0 = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c0 - h, 3), round(c0 + h, 3)
print("4 exploits in 100 audited successes:", wilson(4, 100), "; 0 in 100: upper bound", round(1 - 0.05 ** (1 / 100), 3))
print("computer use, 50 steps at 99% per-step reliability:", round(0.99 ** 50, 3), "; at 99.9%:", round(0.999 ** 50, 3))
print("pass^8 for a 60% agent:", round(0.6 ** 8, 4))
PROTECTED = ("tests/", "test_", "conftest.py", "pytest.ini", "setup.cfg", ".github/")
def screen_patch(changed_files):
    touched = [f for f in changed_files if any(p in f for p in PROTECTED)]
    return {"ok": not touched, "touched_protected": touched}
print(screen_patch(["src/parser.py", "tests/test_parser.py"]))
print("E.10 sizes: 300 tasks x 8 runs x 12 calls =", 300 * 8 * 12, "; 0 over-cap payouts in 2,400 runs -> upper bound",
      round(1 - 0.05 ** (1 / 2400), 5), "; context precision with b at rank 1:", context_precision_at_k(["b", "a", "c", "d"], {"b", "d"}))
