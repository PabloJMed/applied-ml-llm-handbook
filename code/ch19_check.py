"""Chapter 19 · Post-training II: RL from verifiable rewards and reasoning (NumPy, seconds)."""
import math
import numpy as np
from collections import Counter

rng = np.random.default_rng(0)

# 19.3 GRPO advantages for one group, and how each token's gradient weight depends on the loss normalisation
rewards = np.array([1, 0, 0, 1, 1, 0, 0, 0], float)
lengths = np.array([200, 900, 1500, 300, 250, 400, 1200, 2000])
adv_grpo = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)
adv_dr = rewards - rewards.mean()                            # Dr. GRPO: no std division
print("GRPO advantages:", np.round(adv_grpo, 2).tolist())
# sequence-mean loss (original GRPO): each token weighted by A / |o|  -> long wrong answers are penalised less per token
w_seq = adv_grpo / lengths
# token-mean over the whole group (DAPO) or a constant normaliser (Dr. GRPO): weight A / constant
w_tok = adv_grpo / lengths.sum()
for i in (0, 2, 7):
    print(f"sample {i}: reward {rewards[i]:.0f}, {lengths[i]:>4} tokens: per-token weight seq-mean {w_seq[i]*1e3:+.3f}e-3, "
          f"token-mean {w_tok[i]*1e3:+.3f}e-3")


# 19.5 zero-signal groups: fraction of prompts whose group of G samples is all right or all wrong
G = 8
for p in (0.02, 0.1, 0.5, 0.9, 0.98):
    print(f"pass rate {p:.2f}: P(group has no signal) = {p**G + (1-p)**G:.3f}")

# 19.4 PPO-style clipping with a higher upper bound (DAPO clip-higher)
def clipped(ratio, A, lo=0.2, hi=0.2):
    return min(ratio * A, np.clip(ratio, 1 - lo, 1 + hi) * A)
p_old = 0.01                                                    # a rare token that led to a correct answer
for hi in (0.2, 0.28):
    max_ratio = 1 + hi
    print(f"clip high {hi}: a token at p=0.01 can rise to at most p={p_old*max_ratio:.4f} per update")

# 19.6 does RL create new solutions or sharpen existing ones? A tabular simulation.
# 200 problems; each has 6 strategies; strategy s solves problem j with probability q[j, s].
J, S = 200, 6
q = np.zeros((J, S))
q[:, 0] = rng.uniform(0.3, 0.9, J)                     # a common strategy that works on most problems...
q[:, 0] *= rng.random(J) > 0.25                        # ...but fails completely on 25% of them
q[:, 1:] = rng.uniform(0, 0.05, (J, S - 1))
hard = q[:, 0] == 0
q[hard, 5] = 0.8                                       # those problems need a rare strategy
logits = np.log(np.array([0.70, 0.12, 0.08, 0.05, 0.04, 0.01]))[None].repeat(J, 0)   # shared prior...
theta = np.zeros(S)                                    # ...plus a shared learned preference (the "policy")
def policy():
    z = logits + theta; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)
def pass_at_k(pi, k):
    p1 = (pi * q).sum(1)                                # per-problem single-sample success
    return (1 - (1 - p1) ** k).mean()
pi = policy()
print(f"base:  pass@1 {pass_at_k(pi,1):.3f}  pass@64 {pass_at_k(pi,64):.3f}  entropy {-(pi*np.log(pi)).sum(1).mean():.2f}")
for step in range(400):                                 # GRPO-style updates on a batch of problems
    js = rng.integers(0, J, 32); grad = np.zeros(S); pi = policy()
    for j in js:
        acts = rng.choice(S, G, p=pi[j]); r = (rng.random(G) < q[j, acts]).astype(float)
        if r.std() == 0: continue
        A = (r - r.mean()) / r.std()
        for a, adv in zip(acts, A):
            g = -pi[j].copy(); g[a] += 1; grad += adv * g      # d log pi(a) / d theta for a softmax
    theta += 0.05 * grad / len(js)
pi = policy()
print(f"after: pass@1 {pass_at_k(pi,1):.3f}  pass@64 {pass_at_k(pi,64):.3f}  entropy {-(pi*np.log(pi)).sum(1).mean():.2f}")
print(f"hard problems (need the rare strategy): pass@64 base {1-(1-0.01*0.8)**64:.3f} -> after "
      f"{(1-(1-(pi*q).sum(1))**64)[hard].mean():.3f}")

# 19.8 self-consistency (majority vote) vs single sample
def majority_accuracy(p_correct, n, n_wrong_answers=5, trials=20000):
    wins = 0
    for _ in range(trials):
        answers = np.where(rng.random(n) < p_correct, 0, rng.integers(1, n_wrong_answers + 1, n))
        c = Counter(answers.tolist()); top = max(c.values())
        winners = [a for a, v in c.items() if v == top]
        wins += (0 in winners) / len(winners)
    return wins / trials
for p in (0.3, 0.5):
    print(f"p={p}: majority of 1 {majority_accuracy(p,1):.3f}, of 5 {majority_accuracy(p,5):.3f}, of 25 {majority_accuracy(p,25):.3f}")
print("p=0.3 with a single wrong answer taking the other 70%: majority of 25 = %.3f" % majority_accuracy(0.3, 25, 1))

# 19.11 share of groups with no signal, by pass rate and group size
for G in (8, 16):
    print(f"G={G}:", {q: round(q ** G + (1 - q) ** G, 3) for q in (0.05, 0.2, 0.5)})
print("generated tokens per step (512 prompts x 16 x 4k): %.1fM" % (512 * 16 * 4000 / 1e6))
