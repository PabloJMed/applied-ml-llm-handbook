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
    df["roll_mean_7d_same_hour"] = (df.groupby(["zone", "hour"])["orders"]
                                      .transform(lambda s: s.shift(7).rolling(4, min_periods=1).mean()))
    return df
feat = add_features(df).dropna()
cutoff = pd.Timestamp("2026-04-15")
train, test = feat[feat.ts < cutoff], feat[feat.ts >= cutoff]
cols = ["zone", "hour", "dow", "lag_168", "lag_336", "roll_mean_7d_same_hour"]

def wape(y, p): return np.abs(y - p).sum() / np.abs(y).sum()
print("seasonal naive WAPE", round(wape(test.orders, test.lag_168), 3))
m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(train[cols], train.orders)
print("LightGBM WAPE", round(wape(test.orders, m.predict(test[cols])), 3))
q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=400, learning_rate=0.05, verbose=-1).fit(train[cols], train.orders)
print("P90 coverage", round(np.mean(test.orders <= q90.predict(test[cols])), 3))

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
