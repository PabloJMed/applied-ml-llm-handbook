import sqlite3, math, pandas as pd, numpy as np
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

# ---- added: interview SQL, RANK vs DENSE_RANK, ROWS vs RANGE, cohort retention, kappa arithmetic ----
uo=pd.DataFrame({"order_id":range(1,10),"user_id":[1,1,1,1,2,2,3,3,3],
 "order_ts":["2026-01-20 12:00","2026-02-10 19:30","2026-02-25 13:00","2026-03-01 20:15","2025-12-30 18:00","2026-02-28 12:45","2026-02-01 09:00","2026-02-20 21:00","2026-02-20 21:00"],
 "amount":[25,40,18,32,55,22,15,30,12]})
uo.to_sql("user_orders",con,index=False)
q1="""
WITH ranked AS (
  SELECT user_id, order_id, order_ts,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_ts DESC, order_id DESC) AS rn
  FROM user_orders),
recent AS (
  SELECT user_id, COUNT(*) AS orders_30d
  FROM user_orders
  WHERE order_ts >= datetime('2026-03-02', '-30 days') AND order_ts < '2026-03-02'
  GROUP BY user_id)
SELECT r.user_id, r.order_id AS latest_order_id, r.order_ts AS latest_order_ts,
       COALESCE(c.orders_30d, 0) AS orders_30d
FROM ranked r LEFT JOIN recent c ON c.user_id = r.user_id
WHERE r.rn = 1
ORDER BY r.user_id"""
print(pd.read_sql(q1,con))
sc=pd.DataFrame({"restaurant":["A","B","C","D","E"],"rating":[4.8,4.6,4.6,4.5,4.4]})
sc.to_sql("ratings",con,index=False)
q2="""SELECT restaurant, rating,
 ROW_NUMBER() OVER (ORDER BY rating DESC) AS row_number,
 RANK() OVER (ORDER BY rating DESC) AS rank,
 DENSE_RANK() OVER (ORDER BY rating DESC) AS dense_rank
FROM ratings"""
print(pd.read_sql(q2,con))
# rows vs range ties
o=pd.DataFrame({"order_id":[1,2,3,4],"restaurant_id":[7,7,7,7],"order_ts":["2026-03-01","2026-03-02","2026-03-02","2026-03-05"],"amount":[50,30,20,80]})
o.to_sql("o",con,index=False)
q3="""SELECT order_id, order_ts, amount,
 SUM(amount) OVER (PARTITION BY restaurant_id ORDER BY order_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS rows_before,
 SUM(amount) OVER (PARTITION BY restaurant_id ORDER BY julianday(order_ts) RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS range_before
FROM o ORDER BY order_ts, order_id"""
print(pd.read_sql(q3,con))
# cohort retention
co=pd.DataFrame({"user_id":[1,1,1,2,2,3,3,3,4,5,5],
 "order_ts":["2026-01-05","2026-02-11","2026-03-20","2026-01-17","2026-03-02","2026-01-25","2026-02-03","2026-02-27","2026-02-08","2026-02-14","2026-03-09"]})
co.to_sql("orders_c",con,index=False)
q4="""
WITH first AS (
  SELECT user_id, MIN(strftime('%Y-%m', order_ts)) AS cohort
  FROM orders_c GROUP BY user_id),
activity AS (
  SELECT DISTINCT o.user_id, f.cohort,
         (CAST(strftime('%Y', o.order_ts) AS INT) - CAST(substr(f.cohort, 1, 4) AS INT)) * 12
       + (CAST(strftime('%m', o.order_ts) AS INT) - CAST(substr(f.cohort, 6, 2) AS INT)) AS month_offset
  FROM orders_c o JOIN first f USING (user_id)),
sizes AS (SELECT cohort, COUNT(*) AS cohort_size FROM first GROUP BY cohort)
SELECT a.cohort, a.month_offset, COUNT(*) AS active_users, s.cohort_size,
       ROUND(1.0 * COUNT(*) / s.cohort_size, 2) AS retention
FROM activity a JOIN sizes s USING (cohort)
GROUP BY a.cohort, a.month_offset
ORDER BY a.cohort, a.month_offset"""
print(pd.read_sql(q4,con))
# kappa
a = np.array(list("PPNPNNPPNP")+["P"]*10); b = np.array(list("PNNPNPPPNP")+["P"]*10)
pa=(a=="P").mean(); pb=(b=="P").mean(); po=(a==b).mean(); pe=pa*pb+(1-pa)*(1-pb)
print("pa pb",pa,pb,"po",po,"pe",pe,"kappa",(po-pe)/(1-pe), cohen_kappa_score(a,b))
# weighted kappa example 0-3 ratings
r1=np.array([3,3,2,2,1,0,3,2,1,1]); r2=np.array([3,2,2,1,1,0,2,2,0,1])
print("unweighted",round(cohen_kappa_score(r1,r2),3),"linear",round(cohen_kappa_score(r1,r2,weights="linear"),3),"quadratic",round(cohen_kappa_score(r1,r2,weights="quadratic"),3), "raw",(r1==r2).mean())
# preference agreement -> single rater accuracy

for agree in (0.6,0.65,0.7,0.73,0.75):
    q=(1+math.sqrt(1-2*(1-agree)))/2
    print(agree, "single-rater match to consensus truth", round(q,3))
# labelling budget
print("labelling hours", 10000 * 30 / 3600, "QC", 2000 * 30 / 3600)
