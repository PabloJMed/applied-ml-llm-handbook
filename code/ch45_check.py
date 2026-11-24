import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import linear_sum_assignment, linprog
rng = np.random.default_rng(0)

# synthetic hourly demand for 5 zones over 120 days
hours = pd.date_range("2026-01-01", periods=24 * 120, freq="h")
rows = []
for z in range(5):
    base = 20 + 10 * z
    daily = np.where((hours.hour >= 12) & (hours.hour <= 14), 2.0, np.where((hours.hour >= 19) & (hours.hour <= 22), 2.6, 0.6))
    weekly = np.where(hours.dayofweek >= 4, 1.25, 1.0)
    lam = base * daily * weekly * (1 + 0.002 * np.arange(len(hours)) / 24)
    rows.append(pd.DataFrame({"ts": hours, "zone": z, "orders": rng.poisson(lam)}))
df = pd.concat(rows).sort_values(["zone", "ts"]).reset_index(drop=True)

def add_features(df):
    df = df.copy()
    df["hour"] = df.ts.dt.hour; df["dow"] = df.ts.dt.dayofweek
    g = df.groupby("zone")["orders"]
    for lag in (24 * 7, 24 * 14):                      # same hour last week and two weeks ago
        df[f"lag_{lag}"] = g.shift(lag)
    # same weekday and hour over the previous four weeks (shift(1) within the group = 7 days back)
    df["roll_mean_same_dow_hour"] = (df.groupby(["zone", "dow", "hour"])["orders"]
                                       .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()))
    return df
feat = add_features(df).dropna()
cutoff = pd.Timestamp("2026-04-15")
train, test = feat[feat.ts < cutoff], feat[feat.ts >= cutoff]
cols = ["zone", "hour", "dow", "lag_168", "lag_336", "roll_mean_same_dow_hour"]

def wape(y, p): return np.abs(y - p).sum() / np.abs(y).sum()
print("seasonal naive WAPE", round(wape(test.orders, test.lag_168), 3))
m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=2).fit(train[cols], train.orders)
print("LightGBM WAPE", round(wape(test.orders, m.predict(test[cols])), 3))
q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=400, learning_rate=0.05, verbose=-1, n_jobs=2).fit(train[cols], train.orders)
print("P90 coverage", round(np.mean(test.orders <= q90.predict(test[cols])), 3))

# split-conformal (CQR-style) correction of the P90: calibrate on the last two weeks of training data
cal_start = cutoff - pd.Timedelta(days=14)
fit, cal = train[train.ts < cal_start], train[train.ts >= cal_start]
q90c = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=400, learning_rate=0.05, verbose=-1, n_jobs=2).fit(fit[cols], fit.orders)
scores = cal.orders.values - q90c.predict(cal[cols])            # positive = under-covered
n_cal, alpha_c = len(scores), 0.1
k = np.quantile(scores, min(1.0, (1 - alpha_c) * (1 + 1 / n_cal)), method="higher")
print("conformal shift", round(k, 2), "n_cal", n_cal,
      "raw coverage", round(np.mean(test.orders <= q90c.predict(test[cols])), 3),
      "conformal coverage", round(np.mean(test.orders <= q90c.predict(test[cols]) + k), 3))

# assignment: couriers to orders minimising pickup time
cour = rng.uniform(0, 5, (6, 2)); orders = rng.uniform(0, 5, (5, 2))
cost = np.linalg.norm(cour[:, None, :] - orders[None], axis=2) / 0.25   # minutes at 15 km/h
r, c = linear_sum_assignment(cost)
print(list(zip(r, c)), round(cost[r, c].sum(), 1))
greedy_total, used = 0, set()
for j in range(len(orders)):
    i = min((i for i in range(len(cour)) if i not in used), key=lambda i: cost[i, j]); used.add(i); greedy_total += cost[i, j]
print("greedy", round(greedy_total, 1))

# LP rebalancing: move idle couriers from surplus to deficit zones at min travel cost
supply = np.array([12, 3, 8, 2])     # idle couriers now
need = np.array([6, 7, 5, 7])        # forecast need next 30 min
dist = np.array([[0, 4, 6, 8], [4, 0, 3, 5], [6, 3, 0, 4], [8, 5, 4, 0]], float)
n = len(supply)
c_vec = dist.flatten()
A_ub, b_ub = [], []
for i in range(n):                    # can't move more than you have
    row = np.zeros(n * n); row[i * n:(i + 1) * n] = 1; A_ub.append(row); b_ub.append(supply[i])
for j in range(n):                    # each zone ends with at least its need
    row = np.zeros(n * n); row[j::n] = -1; A_ub.append(row); b_ub.append(-need[j])
res = linprog(c_vec, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=(0, None), method="highs")
print(res.status, np.round(res.x.reshape(n, n)), res.fun)

# --- 45.6 Erlang C staffing (M/M/c): Poisson orders, exponential 25-minute courier time per order ---
import math
from scipy import stats

def erlang_c(c, a):
    """Probability an arriving order has to wait, c servers, offered load a = lambda/mu (needs c > a)."""
    b = 1.0
    for k in range(1, c + 1):          # Erlang B by the stable recursion, then convert to Erlang C
        b = a * b / (k + a * b)
    return c * b / (c - a * (1 - b))

def p_late(lam_per_h, c, svc_min=25, t_min=5):
    """P(an order waits more than t_min for a courier); every order is late if c <= load."""
    mu = 60 / svc_min
    a = lam_per_h / mu
    if c <= a:
        return 1.0
    return erlang_c(c, a) * math.exp(-(c * mu - lam_per_h) * t_min / 60)

for lam in (100, 120, 140):
    a = lam / 2.4
    c = math.ceil(a)
    while 1 - p_late(lam, c) < 0.9:
        c += 1
    print(f"{lam}/h load {a:.1f} -> {c} couriers, service level {1 - p_late(lam, c):.3f}, "
          f"P(wait) {erlang_c(c, a):.3f}, beta {(c - a) / math.sqrt(a):.2f}")

# newsvendor step: expected cost over demand ~ Normal(120, 15), EUR 6 per late order, EUR 9 per courier-hour
xs = np.arange(40, 221); w = stats.norm.pdf(xs, 120, 15); w /= w.sum()
def late_orders(c): return sum(wi * x * p_late(x, c) for x, wi in zip(xs, w))
cost = {c: 6 * late_orders(c) + 9 * c for c in range(50, 75)}
best = min(cost, key=cost.get)
print("cost at 56", round(cost[56]), "best", best, round(cost[best]), "64-66", [round(cost[c], 1) for c in (64, 65, 66)])
for c in (63, 64, 65, 66):
    saved = late_orders(c) - late_orders(c + 1)
    print(f"courier {c + 1}: avoids {saved:.2f} late orders/h = EUR {6 * saved:.2f} vs EUR 9")
print("P90 demand", round(stats.norm.ppf(0.9, 120, 15), 1))
for lam in (130, 140, 150):
    print(f"at {lam}/h the 65th courier avoids {lam * (p_late(lam, 64) - p_late(lam, 65)):.1f} late orders")

# --- 45.4 elasticity: switchback estimate vs naive observational regression ---
sess_a, conv_a, fee_a = 600, 0.200, 2.99
conv_b, fee_b = 0.190, 3.49
dq, dp = (conv_b - conv_a) / conv_a, (fee_b - fee_a) / fee_a
print(f"elasticity {dq / dp:.2f} (dq {dq:.3f}, dp {dp:.3f}); orders {sess_a * conv_a:.0f} -> {sess_a * conv_b:.0f}; "
      f"fee revenue {sess_a * conv_a * fee_a:.0f} -> {sess_a * conv_b * fee_b:.0f}; "
      f"courier load {sess_a * conv_a / 2.4:.1f} -> {sess_a * conv_b / 2.4:.1f}")
# observational data: surge raises the fee exactly when underlying demand is high
rng2 = np.random.default_rng(1)
demand_index = rng2.lognormal(0, 0.35, 5000)                 # busy vs quiet hours
fee = 2.99 * (1 + 0.4 * (demand_index > 1.3))                 # surge pricing when busy
orders_obs = rng2.poisson(120 * demand_index * (fee / 2.99) ** (dq / dp))
naive = np.polyfit(np.log(fee), np.log(orders_obs + 1), 1)[0]
print(f"naive log-log slope {naive:.2f} vs true elasticity {dq / dp:.2f}")
print("service level with 55 couriers at 120/h", round(1 - p_late(120, 55), 3))
c = 101
while 1 - p_late(240, c) < 0.9:
    c += 1
print("pooled zone, 240/h:", c, "couriers vs 2 x 56 = 112 separately")
