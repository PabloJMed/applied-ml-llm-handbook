import numpy as np, pandas as pd, sqlite3
from dataclasses import dataclass

orders = pd.DataFrame({
    "order_id": range(1, 11),
    "user_id": [1, 1, 2, 2, 2, 3, 3, 1, 4, 4],
    "restaurant": ["a", "b", "a", "a", "c", "b", "b", "a", "c", "a"],
    "ts": pd.to_datetime(["2026-10-01 12:00", "2026-10-01 12:20", "2026-10-02 19:00", "2026-10-05 20:00",
                          "2026-10-09 13:00", "2026-10-03 21:00", "2026-10-03 21:10", "2026-10-15 12:30",
                          "2026-10-07 18:00", "2026-10-20 19:30"]),
    "amount": [18.5, 22.0, 15.0, 31.0, 12.5, 27.0, 9.0, 20.0, 14.0, 25.5],
})

# 1 top-k per group
top2 = (orders.sort_values(["user_id", "amount"], ascending=[True, False])
              .groupby("user_id").head(2))
print(top2[["user_id", "amount"]].values.tolist())

# 2 share of each user's spend by restaurant (transform)
spend = orders.groupby(["user_id", "restaurant"], as_index=False)["amount"].sum()
spend["share"] = spend["amount"] / spend.groupby("user_id")["amount"].transform("sum")
print(spend.round(2).values.tolist()[:3])

# 3 days since previous order (shift within group)
o = orders.sort_values(["user_id", "ts"]).copy()
o["days_since_prev"] = o.groupby("user_id")["ts"].diff().dt.total_seconds() / 86400
print(o[["user_id", "days_since_prev"]].round(2).values.tolist()[:4])

# 4 sessionisation: new session if gap > 30 min
o["new_session"] = (o["ts"].diff() > pd.Timedelta("30min")) | (o["user_id"] != o["user_id"].shift())
o["session_id"] = o["new_session"].cumsum()
print(o.groupby("user_id")["session_id"].nunique().to_dict())

# 5 rolling 7-day spend per user (time-based window)
roll = (o.set_index("ts").groupby("user_id")["amount"].rolling("7D").sum().reset_index(name="spend_7d"))
print(roll.values.tolist()[:3])

# 6 point-in-time join
feats = pd.DataFrame({"restaurant": ["a", "a", "b", "c"], "ts": pd.to_datetime(["2026-09-30", "2026-10-06", "2026-09-30", "2026-10-01"]), "rating": [4.2, 4.5, 3.9, 4.8]}).sort_values("ts")
pit = pd.merge_asof(orders.sort_values("ts"), feats, on="ts", by="restaurant", direction="backward")
print(pit[["order_id", "restaurant", "rating"]].sort_values("order_id").values.tolist()[:5])

# 7 retention: share of users ordering again within 7 days of first order
first = orders.groupby("user_id")["ts"].min().rename("first_ts")
m = orders.join(first, on="user_id")
returned = m[(m.ts > m.first_ts) & (m.ts <= m.first_ts + pd.Timedelta("7D"))].user_id.unique()
print("7d repeat rate", len(returned) / orders.user_id.nunique())

# SQL with window functions in sqlite
con = sqlite3.connect(":memory:")
orders.assign(ts=orders.ts.astype(str)).to_sql("orders", con, index=False)
q = """
WITH ranked AS (
  SELECT user_id, order_id, amount, ts,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts) AS n,
         LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) AS prev_ts,
         SUM(amount) OVER (PARTITION BY user_id ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_spend
  FROM orders)
SELECT user_id, order_id, n, prev_ts, running_spend FROM ranked WHERE user_id = 2 ORDER BY ts;
"""
print(con.execute(q).fetchall())
q2 = """
WITH firsts AS (SELECT user_id, MIN(ts) AS first_ts FROM orders GROUP BY user_id)
SELECT COUNT(DISTINCT o.user_id) * 1.0 / (SELECT COUNT(*) FROM firsts) AS repeat_7d
FROM orders o JOIN firsts f ON o.user_id = f.user_id
WHERE o.ts > f.first_ts AND julianday(o.ts) <= julianday(f.first_ts) + 7;
"""
print(con.execute(q2).fetchall())

# OOP: sklearn-compatible estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

class LogisticRegressionGD(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.1, n_iter=500, l2=0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        Xb = np.c_[np.ones(len(X)), X]
        self.w_ = np.zeros(Xb.shape[1])
        for _ in range(self.n_iter):
            p = 1 / (1 + np.exp(-Xb @ self.w_))
            grad = Xb.T @ (p - y) / len(y)
            grad[1:] += self.l2 * self.w_[1:]
            self.w_ -= self.lr * grad
        return self
    def predict_proba(self, X):
        check_is_fitted(self, "w_")
        X = check_array(X)
        p = 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ self.w_)))
        return np.c_[1 - p, p]
    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

X, y = make_classification(n_samples=500, n_features=5, random_state=0)
print(cross_val_score(LogisticRegressionGD(), X, y, cv=5).mean())

def test_predict_proba_sums_to_one():
    m = LogisticRegressionGD().fit(X, y)
    assert np.allclose(m.predict_proba(X).sum(1), 1)
def test_learns_separable():
    Xs = np.r_[np.random.default_rng(0).normal(-2, 1, (50, 2)), np.random.default_rng(1).normal(2, 1, (50, 2))]
    ys = np.r_[np.zeros(50), np.ones(50)]
    assert (LogisticRegressionGD().fit(Xs, ys).predict(Xs) == ys).mean() > 0.95
test_predict_proba_sums_to_one(); test_learns_separable(); print("tests ok")
