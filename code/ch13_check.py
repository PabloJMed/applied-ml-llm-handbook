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
