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
