import sqlite3, pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score
con = sqlite3.connect(":memory:")
events = pd.DataFrame({"user_id":[1,1,1,1,2,2],"ts":pd.to_datetime(["2026-03-01 10:00","2026-03-01 10:10","2026-03-01 11:00","2026-03-01 11:05","2026-03-01 09:00","2026-03-01 09:50"]).astype(str)})
events.to_sql("events", con, index=False)
q_sess = """
WITH g AS (
  SELECT user_id, ts,
         CASE WHEN (julianday(ts) - julianday(LAG(ts) OVER (PARTITION BY user_id ORDER BY ts))) * 1440 > 30
                OR LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) IS NULL THEN 1 ELSE 0 END AS new_session
  FROM events)
SELECT user_id, ts, SUM(new_session) OVER (PARTITION BY user_id ORDER BY ts) AS session_id FROM g ORDER BY user_id, ts"""
print(pd.read_sql(q_sess, con))
orders = pd.DataFrame({"order_id":[1,2,3,4,5],"restaurant_id":[7,7,7,8,8],"order_ts":["2026-03-01","2026-03-02","2026-03-05","2026-03-02","2026-03-03"],"amount":[50,30,80,20,40]})
orders.to_sql("orders", con, index=False)
q_win = """SELECT order_id, restaurant_id, order_ts, amount,
  ROW_NUMBER() OVER (PARTITION BY restaurant_id ORDER BY order_ts) AS rn,
  SUM(amount) OVER (PARTITION BY restaurant_id ORDER BY order_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS spend_before,
  AVG(amount) OVER (PARTITION BY restaurant_id ORDER BY order_ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS avg_last2
FROM orders ORDER BY restaurant_id, order_ts"""
print(pd.read_sql(q_win, con))
# point in time join
labels = pd.DataFrame({"restaurant_id":[7,7,8],"ts":pd.to_datetime(["2026-03-02 12:00","2026-03-06 12:00","2026-03-02 12:00"])}).sort_values("ts")
feats = pd.DataFrame({"restaurant_id":[7,7,7,8],"feature_ts":pd.to_datetime(["2026-03-01","2026-03-03","2026-03-07","2026-03-03"]),"rating_7d":[4.1,4.3,3.9,4.6]}).sort_values("feature_ts")
pit = pd.merge_asof(labels, feats, left_on="ts", right_on="feature_ts", by="restaurant_id", direction="backward")
print(pit)
naive = labels.merge(feats.sort_values("feature_ts").groupby("restaurant_id").tail(1), on="restaurant_id")
print(naive[["restaurant_id","ts","rating_7d"]])
# kappa
a = np.array(list("PPNPNNPPNP")+["P"]*10); b = np.array(list("PNNPNPPPNP")+["P"]*10)
print("agree", (a==b).mean(), "kappa", round(cohen_kappa_score(a,b),3))
# data contract
def check_contract(df):
    errs=[]
    if df["amount"].isna().mean()>0.01: errs.append("amount nulls")
    if (df["amount"]<0).any(): errs.append("negative amount")
    if df["order_id"].duplicated().any(): errs.append("duplicate order_id")
    return errs
bad = orders.copy(); bad.loc[5]=[5,8,"2026-03-04",-3]
print(check_contract(orders), check_contract(bad))
