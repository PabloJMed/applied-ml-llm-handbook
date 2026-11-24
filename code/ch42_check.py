# serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np, time

app = FastAPI(title="eta-model")

class EtaRequest(BaseModel):
    distance_km: float = Field(ge=0, le=50)
    prep_minutes: float = Field(ge=0, le=120)
    is_rush_hour: bool

class EtaResponse(BaseModel):
    eta_minutes: float
    model_version: str

W = np.array([2.1, 1.0, 6.0]); B = 4.0          # stand-in for a trained model loaded at startup
MODEL_VERSION = "eta-2026-11-01"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=EtaResponse)
def predict(req: EtaRequest):
    x = np.array([req.distance_km, req.prep_minutes, float(req.is_rush_hour)])
    return EtaResponse(eta_minutes=round(float(W @ x + B), 1), model_version=MODEL_VERSION)

# ---- test it without a network: TestClient calls the app in-process
from fastapi.testclient import TestClient
client = TestClient(app)
print(client.get("/health").json())
r = client.post("/predict", json={"distance_km": 3.2, "prep_minutes": 12, "is_rush_hour": True})
print(r.status_code, r.json())
bad = client.post("/predict", json={"distance_km": -1, "prep_minutes": 12, "is_rush_hour": True})
print(bad.status_code, bad.json()["detail"][0]["msg"])

lat = []
for _ in range(2000):
    t = time.perf_counter(); client.post("/predict", json={"distance_km": 3.2, "prep_minutes": 12, "is_rush_hour": False}); lat.append((time.perf_counter()-t)*1000)
lat = np.array(lat); print("p50 %.2f ms  p95 %.2f ms  p99 %.2f ms" % tuple(np.percentile(lat, [50, 95, 99])))

# Little's law sizing
def replicas_needed(peak_rps, latency_s, concurrency_per_replica, headroom=0.6):
    in_flight = peak_rps * latency_s                 # L = lambda * W
    return int(np.ceil(in_flight / (concurrency_per_replica * headroom)))
print(replicas_needed(peak_rps=3000, latency_s=0.05, concurrency_per_replica=8))
print(replicas_needed(peak_rps=3000, latency_s=0.40, concurrency_per_replica=8))

# ---------------------------------------------------------------------------
# Readiness vs liveness (42.5)
from contextlib import asynccontextmanager
from fastapi import Response
STATE = {"model": None}
@asynccontextmanager
async def lifespan(app_):
    STATE["model"] = object(); yield
app2 = FastAPI(lifespan=lifespan)
@app2.get("/health")
def health2():
    return {"status": "ok"}
@app2.get("/ready")
def ready(response: Response):
    ok = STATE["model"] is not None
    response.status_code = 200 if ok else 503
    return {"ready": ok}
c2 = TestClient(app2)
print("ready before start-up:", c2.get("/ready").status_code)          # lifespan not run yet
with TestClient(app2) as c3:
    print("ready after start-up:", c3.get("/ready").status_code, "health:", c3.get("/health").status_code)

# Capacity arithmetic
import math
print("Q7 replicas", replicas_needed(2000, 0.08, 4), "with doubled latency", replicas_needed(2000, 0.16, 4))
print("practice 5:", [replicas_needed(3000, w, 8) for w in (0.05, 0.10, 0.40)])
print("HPA 6 pods at 90% CPU, target 60%:", math.ceil(6 * 90 / 60))
for rho in (0.5, 0.8, 0.9):
    print(f"M/M/1, 50 ms mean service: utilisation {rho}: p99 {math.log(100) / (20 * (1 - rho)):.2f} s")
print("share of page loads with >=1 call beyond p99, 20 calls:", round(1 - 0.99 ** 20, 3))
print("queue backlog after 10 min at 1000/s in, 600/s out:", 400 * 600, "drain minutes at 300/s spare:", round(240000 / 300 / 60, 1))
print("weekly LLM job cost real time:", round(10e6 * (500 * 0.15e-6 + 100 * 0.6e-6)), "hours at 2M TPM:", 10e6 * 600 / 2e6 / 60)
print("LLM replicas:", math.ceil(50 * 300 / (2500 * 0.7)), "70B bf16 GB:", 70e9 * 2 / 1e9, "download s at 1 GB/s:", 140)
print("70B checkpoint TB at 14 B/param:", 70e9 * 14 / 1e12, "seconds at 20 GB/s:", 70e9 * 14 / 20e9)

# Research exercise: fan-out tail and hedging, simulated with log-normal backend latencies
rng = np.random.default_rng(0)
def backend(n):
    return rng.lognormal(mean=np.log(20), sigma=0.6, size=n)           # ms
ref = backend(1_000_000); p95, p99 = np.percentile(ref, [95, 99])
for N in (1, 10, 100):
    plain = backend((50_000, N)).max(1)
    first = backend((50_000, N)); second = p95 + backend((50_000, N))     # backup sent at the p95
    hedged = np.where(first > p95, np.minimum(first, second), first).max(1)
    print(f"fan-out {N:3d}: share slower than backend p99 {np.mean(plain > p99):.3f}; with hedging at p95 {np.mean(hedged > p99):.3f}")
