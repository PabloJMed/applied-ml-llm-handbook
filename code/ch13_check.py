"""Chapter 13 · Scaling laws: runnable checks (NumPy + SciPy, seconds)."""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)

# 13.3 Chinchilla parametric fit (Hoffmann et al. 2022, approach 3)
E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
L = lambda N, D: E + A / N**alpha + B / D**beta

def compute_optimal(C):
    """Minimise L(N, D) subject to C = 6 N D (closed form)."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

for C in (1e21, 1e23, 1e25):
    N, D = compute_optimal(C)
    print(f"C={C:.0e}: N_opt={N/1e9:6.2f}B  D_opt={D/1e9:8.0f}B tokens  D/N={D/N:5.1f}  loss={L(N, D):.3f}")
print("exponents: N_opt ~ C^%.2f, D_opt ~ C^%.2f" % (beta / (alpha + beta), alpha / (alpha + beta)))
print("Chinchilla 70B/1.4T loss %.3f   Gopher 280B/300B loss %.3f" % (L(70e9, 1.4e12), L(280e9, 300e9)))

# 13.4 the IsoFLOP method on synthetic runs: for each budget, fit a parabola in log N and take its minimum
budgets = [1e18, 3e18, 1e19, 3e19, 1e20]
mins = []
for C in budgets:
    Ns = np.logspace(np.log10(compute_optimal(C)[0]) - 0.8, np.log10(compute_optimal(C)[0]) + 0.8, 9)
    losses = np.array([L(N, C / (6 * N)) for N in Ns]) * np.exp(rng.normal(0, 0.003, len(Ns)))  # 0.3% noise
    a2, a1, a0 = np.polyfit(np.log(Ns), losses, 2)
    mins.append(np.exp(-a1 / (2 * a2)))
slope, intercept = np.polyfit(np.log(budgets), np.log(mins), 1)
print(f"IsoFLOP fit: N_opt ~ C^{slope:.2f} (true {beta/(alpha+beta):.2f}); extrapolated N_opt at 1e23: "
      f"{np.exp(intercept) * 1e23**slope / 1e9:.1f}B (true {compute_optimal(1e23)[0]/1e9:.1f}B)")

# 13.4 fitting the parametric form itself (Huber loss on log-loss, as in the paper)
runs = [(N, D) for N in np.logspace(7.5, 9.5, 6) for D in np.logspace(9, 11, 6)]
obs = np.array([L(N, D) for N, D in runs]) * np.exp(rng.normal(0, 0.005, len(runs)))
def objective(p):
    e, a, b, al, be = p
    pred = np.logaddexp.reduce([np.full(len(runs), e),
                                a - al * np.log([r[0] for r in runs]),
                                b - be * np.log([r[1] for r in runs])], axis=0)
    r = pred - np.log(obs)
    d = 1e-3
    return np.where(np.abs(r) < d, 0.5 * r**2, d * (np.abs(r) - 0.5 * d)).sum()
best = min((minimize(objective, x0, method="L-BFGS-B") for x0 in
            [[0.5, 6, 6, 0.3, 0.3], [0.0, 5, 5, 0.4, 0.2], [1.0, 7, 7, 0.2, 0.4]]), key=lambda r: r.fun)
e, a, b, al, be = best.x
print(f"refit: E={np.exp(e):.2f} A={np.exp(a):.0f} B={np.exp(b):.0f} alpha={al:.2f} beta={be:.2f} "
      f"-> exponent N_opt ~ C^{be/(al+be):.2f}")

# 13.5 over-training: minimise training + inference FLOPs to reach a target loss
def total_cost(N, target, inference_tokens):
    D_needed = (B / (target - E - A / N**alpha)) ** (1 / beta)     # tokens needed to hit the target at size N
    return 6 * N * D_needed + 2 * N * inference_tokens, D_needed
target = L(*compute_optimal(1e23))
for inf_tokens in (0, 1e12, 1e13, 1e14):
    Ns = np.logspace(9.3, 11, 400)
    feasible = [N for N in Ns if target - E - A / N**alpha > 0]
    costs = [total_cost(N, target, inf_tokens)[0] for N in feasible]
    N_best = feasible[int(np.argmin(costs))]
    D_best = total_cost(N_best, target, inf_tokens)[1]
    print(f"lifetime inference {inf_tokens:.0e} tokens: N={N_best/1e9:5.1f}B, D={D_best/1e12:5.2f}T, D/N={D_best/N_best:6.0f}")

# 13.6 repeated data: effective tokens (Muennighoff et al. 2023), R* ~ 15.4
U = 100e9
for epochs in (1, 2, 4, 8, 16, 40):
    R = epochs - 1
    eff = U + U * 15.4 * (1 - np.exp(-R / 15.4))
    print(f"{epochs:>2} epochs over 100B unique tokens: {epochs*U/1e9:5.0f}B seen, ~{eff/1e9:5.0f}B effective")

# 13.8 'emergence' from a smooth per-token improvement and an all-or-nothing metric
per_token = np.linspace(0.5, 0.99, 8)
print("per-token accuracy:", np.round(per_token, 2).tolist())
print("exact match, 10-token answer:", np.round(per_token**10, 3).tolist())

# 13.9 test-time compute: pass@k from per-sample success rate, and the unbiased estimator
from math import comb
def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
for p in (0.05, 0.3):
    print(f"p={p}: pass@1 {p:.2f}  pass@8 {1-(1-p)**8:.2f}  pass@64 {1-(1-p)**64:.2f}")
print("unbiased pass@10 from 200 samples with 13 correct: %.3f" % pass_at_k(200, 13, 10))

# ---------------------------------------------------------------------------
# Additions (review pass)
# ---------------------------------------------------------------------------

# 13.1 why power laws: Zipf-distributed "skills" (Michaud et al.). Skill k has frequency p_k ~ k^-(a+1);
# a model that has learned the n most frequent skills still pays for the tail, which shrinks as n^-a.
a_skill = 0.5
p_k = np.arange(1, 10**6 + 1, dtype=float) ** (-(a_skill + 1)); p_k /= p_k.sum()
tail = np.cumsum(p_k[::-1])[::-1]
print("quantised skills, a=0.5: unexplained mass after n skills:",
      {n: round(float(tail[n]), 4) for n in (10, 100, 1000, 10000)}, "(ratio per 10x ~ 10^0.5 = 3.16)")

# 13.3 what 6ND leaves out: causal attention adds ~6 * n_layers * T * d FLOPs per trained token
for T_ctx in (2048, 8192, 32768):
    print(f"attention share of training FLOPs, 8B model (32 layers, d=4096) at T={T_ctx}: "
          f"{6 * 32 * T_ctx * 4096 / (6 * 8e9):.1%}")

# 13.3 the Besiroglu et al. (2024) refit of approach 3, and the 20-tokens-per-parameter rule
E2, A2, B2, al2, be2 = 1.82, 482.01, 2085.43, 0.3478, 0.3658
L2 = lambda N, D: E2 + A2 / N**al2 + B2 / D**be2
def compute_optimal2(C):
    G = (al2 * A2 / (be2 * B2)) ** (1 / (al2 + be2))
    N = G * (C / 6) ** (be2 / (al2 + be2))
    return N, C / (6 * N)
for C in (1e21, 1e23, 1e25):
    N, D = compute_optimal2(C)
    print(f"Besiroglu refit C={C:.0e}: N_opt={N/1e9:6.1f}B  D_opt={D/1e9:6.0f}B  D/N={D/N:4.1f}  loss={L2(N, D):.3f}")
print("20 tokens/param at 1e23: N = sqrt(C/120) = %.1fB, D = %.0fB" % (np.sqrt(1e23 / 120) / 1e9, 20 * np.sqrt(1e23 / 120) / 1e9))

# 13.3 / 13.10 converting a loss gain into a compute multiplier along the frontier L(C) = E + K C^-gamma
for name, (e_, a_, b_, al_, be_), co in (("approach 3", (E, A, B, alpha, beta), compute_optimal),
                                         ("Besiroglu", (E2, A2, B2, al2, be2), compute_optimal2)):
    gamma = al_ * be_ / (al_ + be_)
    Lt = e_ + a_ / co(1e23)[0] ** al_ + b_ / co(1e23)[1] ** be_
    mult = {dl: round(((Lt - e_) / (Lt - e_ - dl)) ** (1 / gamma), 2) for dl in (0.01, 0.02, 0.05)}
    print(f"{name}: gamma={gamma:.3f}, loss at 1e23 {Lt:.3f}; compute multiplier for a loss gain of {mult}")

# 13.5 inference-aware sizing with the Besiroglu constants (target = loss of the compute-optimal 1e23 model)
target2 = L2(*compute_optimal2(1e23))
for inf_tokens in (0, 1e12, 1e13, 1e14):
    Ns = np.logspace(9, 11.5, 4000)
    best2 = None
    for N in Ns:
        r = target2 - E2 - A2 / N**al2
        if r <= 0:
            continue
        D_needed = (B2 / r) ** (1 / be2)
        cost = 6 * N * D_needed + 2 * N * inf_tokens
        if best2 is None or cost < best2[0]:
            best2 = (cost, N, D_needed)
    cost, N, D_needed = best2
    print(f"Besiroglu, lifetime inference {inf_tokens:.0e}: N={N/1e9:5.1f}B D={D_needed/1e12:5.2f}T "
          f"D/N={D_needed/N:5.0f} training {6*N*D_needed:.2e} total {cost:.2e} FLOPs")

# 13.4 bootstrap interval on N_opt(1e23) from the IsoFLOP method (resample the runs at each budget)
Ns_by_budget = []
for C in budgets:
    Ns = np.logspace(np.log10(compute_optimal(C)[0]) - 0.8, np.log10(compute_optimal(C)[0]) + 0.8, 9)
    Ns_by_budget.append((C, Ns, np.array([L(N, C / (6 * N)) for N in Ns]) * np.exp(rng.normal(0, 0.003, len(Ns)))))
pt = []
for C, Ns, ls in Ns_by_budget:
    a2_, a1_, _ = np.polyfit(np.log(Ns), ls, 2)
    pt.append(np.exp(-a1_ / (2 * a2_)))
s_, i_ = np.polyfit(np.log(budgets), np.log(pt), 1)
point = np.exp(i_) * 1e23**s_ / 1e9
boot = []
for _ in range(300):
    mins_b = []
    for C, Ns, ls in Ns_by_budget:
        idx = rng.integers(0, len(Ns), len(Ns))
        a2_, a1_, _ = np.polyfit(np.log(Ns[idx]), ls[idx], 2)
        mins_b.append(np.exp(-a1_ / (2 * a2_)))
    s_, i_ = np.polyfit(np.log(budgets), np.log(mins_b), 1)
    boot.append(np.exp(i_) * 1e23**s_ / 1e9)
lo, hi = np.percentile(boot, [5, 95])
print(f"new noisy draw: point estimate {point:.1f}B; bootstrap 90% interval {lo:.1f}B to {hi:.1f}B "
      f"(true {compute_optimal(1e23)[0]/1e9:.1f}B)")

# 13.8 two-step forecast: accuracy as a sigmoid of loss, loss from the compute law
chance, L_mid, width = 0.25, 2.10, 0.05
acc = lambda loss: chance + (1 - chance) / (1 + np.exp((loss - L_mid) / width))
for loss in (2.30, 2.20, 2.10, 2.035, 2.015):
    print(f"loss {loss:.3f} -> predicted 4-choice accuracy {acc(loss):.3f}")

# 13.10 ladder cost and range
ladder = [150e6, 400e6, 1e9, 2.5e9]
flops = [6 * N * 20 * N * m for N in ladder for m in (0.5, 1, 2)]
print(f"ladder: {sum(flops):.2e} FLOPs per mixture; smallest run {min(flops):.2e}, largest {max(flops):.2e}; "
      f"N range {np.log10(ladder[-1]/ladder[0]):.2f} orders, compute range {np.log10(max(flops)/min(flops)):.2f} orders; "
      f"extrapolation to 1e23: {1e23/max(flops):.0f}x")
print("2.5B on 50B tokens: %.2e FLOPs = %.0f GPU-hours at 40%% of 989 TFLOP/s" % (7.5e20, 7.5e20 / (0.4 * 989e12) / 3600))
N_co = compute_optimal2(1e23)[0]
print("compute-optimal %.1fB model serving 1e14 tokens: total %.2e FLOPs" % (N_co / 1e9, 1e23 + 2 * N_co * 1e14))
