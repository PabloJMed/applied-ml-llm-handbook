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
