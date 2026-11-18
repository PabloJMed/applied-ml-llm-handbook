# 42 · Infrastructure primer

Part XI · Applied ML systems (optional track). No prerequisites: this chapter assumes you have trained models in notebooks but never deployed one. Read it before Chapter 21 (serving) and Chapter 47 (MLOps).

## Why it matters

A model in a notebook helps nobody. To be used, it has to run on a computer that is always on, receive requests over a network, return answers fast enough, survive crashes and traffic spikes, and be updated without breaking anything. That is "infrastructure". Data scientists are not expected to be platform engineers, but in senior interviews and in daily work you must speak the vocabulary: what a container is, why Kubernetes restarts things, what p99 latency means, why a batch job and an API are designed differently, and how many machines a model needs. This chapter explains each piece from zero, in the order a request actually travels.

The Python code was run; outputs are quoted. Dockerfiles, Kubernetes files and shell commands are standard examples that were not run here; check the linked docs for current syntax.

## Key concepts

Each line is a one-sentence definition with a link to a good explanation. The sections below go deeper.

- **Server / virtual machine (VM)**: a computer in a data centre that you rent and reach over the network; a VM is a slice of a physical machine. [Cloud computing overview](https://en.wikipedia.org/wiki/Cloud_computing).
- **CPU vs GPU**: a CPU has a few powerful cores for general work; a GPU has thousands of small cores for doing the same maths on many numbers at once, which is what neural networks need. [NVIDIA explainer](https://blogs.nvidia.com/blog/whats-the-difference-between-a-cpu-and-a-gpu/).
- **Shell and Linux**: almost every server runs Linux and is operated through a text terminal. [The Missing Semester](https://missing.csail.mit.edu/) (MIT) teaches the essentials.
- **HTTP and APIs**: programs talk over the network by sending HTTP requests (GET, POST) to URLs and getting responses, usually JSON. An API is the agreed set of URLs and message formats. [MDN: HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview), [REST](https://en.wikipedia.org/wiki/REST).
- **Latency and throughput**: how long one request takes, and how many requests per second the system handles. Reported as percentiles: p50, p95, p99.
- **Container (Docker)**: your code plus everything it needs (Python version, libraries, model file) packed into one image that runs the same on any machine. [What is a container?](https://www.docker.com/resources/what-container/), [Docker get-started guide](https://docs.docker.com/get-started/).
- **Kubernetes**: software that runs many containers across many machines, restarts them when they crash, and adds or removes copies as traffic changes. [Kubernetes overview](https://kubernetes.io/docs/concepts/overview/).
- **Load balancer**: spreads incoming requests across the copies of a service. [Load balancing](https://en.wikipedia.org/wiki/Load_balancing_(computing)).
- **Autoscaling**: automatically changing the number of copies with load. [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/).
- **Object storage**: cheap storage for files of any size (datasets, model weights), addressed by a path like `s3://bucket/models/v3.pt`. [Amazon S3](https://aws.amazon.com/s3/).
- **Databases vs warehouses**: an app database (OLTP) handles many small reads and writes for the live product; a data warehouse handles big analytical queries. [OLTP](https://en.wikipedia.org/wiki/Online_transaction_processing), [Data warehouse](https://en.wikipedia.org/wiki/Data_warehouse).
- **Cache (Redis)**: a very fast in-memory key-value store for data needed in milliseconds, such as precomputed features. [Redis](https://en.wikipedia.org/wiki/Redis).
- **Queue and stream**: a buffer between producers and consumers of events, so neither has to wait for the other. [Message queue](https://en.wikipedia.org/wiki/Message_queue), [Kafka introduction](https://kafka.apache.org/intro).
- **Batch, online and streaming inference**: predict for everything on a schedule; predict per request; predict per event as it arrives.
- **CI/CD**: automatic testing and deployment on every code change. [CI/CD](https://en.wikipedia.org/wiki/CI/CD).
- **Infrastructure as code (Terraform)**: describing servers, databases and permissions in versioned files instead of clicking in a console. [What is Terraform](https://developer.hashicorp.com/terraform/intro).
- **Observability**: logs, metrics and traces that let you see what a running system is doing. [What is OpenTelemetry?](https://opentelemetry.io/docs/what-is-opentelemetry/), [Prometheus overview](https://prometheus.io/docs/introduction/overview/).
- **Little's law**: the average number of requests in flight equals arrival rate times time in the system; it sizes capacity. [Little's law](https://en.wikipedia.org/wiki/Little%27s_law).

## 42.1 The big picture

![Diagram](images/13a-01.png)

*Most ML systems combine a fast online path (answer one request now) with a slow offline path (process yesterday's data). Read it left to right, top then bottom.*

Two paths cover almost every ML system you will meet:

- **Online path.** A customer opens the app; the app sends a request; within about 100–300 ms the system must answer (a ranked list, an ETA, a fraud score). Everything here is about speed and reliability.
- **Offline path.** Every event (orders, clicks, ratings) is stored. Scheduled jobs process it hours later: build training data, retrain the model, compute features, or score everything in bulk. Everything here is about correctness, cost and volume.

The two meet in two places: the model trained offline is deployed to the online model server, and every online prediction is logged so it becomes offline data. When you design a system (Chapter 46), first decide which parts must be online. Many teams make everything online by default and pay for it in cost and complexity; if a prediction can be computed overnight and looked up, it usually should be.

## 42.2 Machines: CPU, GPU, memory and the cloud

**What a server is.** A computer with CPUs, memory (RAM), disk and a network card, sitting in a data centre. You rarely own one: you rent **virtual machines** from a cloud provider (AWS, Google Cloud, Azure), paying per hour or second.

**CPU vs GPU in ML terms.**

|  | CPU | GPU |
|---|---|---|
| Cores | 4–128 strong, general cores | thousands of simple cores |
| Good at | branching logic, data processing, small models (trees, logistic regression), small batches | large matrix multiplications: training and serving neural networks, especially LLMs |
| Memory | system RAM, tens to hundreds of GB | its own VRAM (for example 24, 48 or 80 GB); the model must fit here |
| Cost | cents to a few dollars per hour | roughly one to several dollars per hour per GPU, more for top models |

Rules of thumb: tree models and small classifiers serve happily on CPUs; transformers serve faster on GPUs; LLMs effectively require GPUs (or a provider's API). The question "does the model fit in GPU memory?" decides a lot; Chapter 21 shows the arithmetic.

**Cloud service levels.** From most control to least: *IaaS* (rent VMs and manage everything), *PaaS / managed services* (the provider runs the database, the Kubernetes cluster or the training job; for ML: SageMaker, Vertex AI, Azure ML, Databricks), *SaaS / APIs* (you just call an endpoint, such as an LLM API). Managed services cost more per hour but less in engineering time; most companies mix them.

**Regions and zones.** A region is a geographic location (for example Frankfurt or Bahrain); zones are separate data centres within it. Run copies in several zones so one data-centre failure does not take you down, and keep data in the region your users and data-protection rules require.

## 42.3 Talking over the network: HTTP and APIs

When the app asks for an ETA, it sends an **HTTP request**: a method (`POST`), a URL (`https://api.example.com/eta/predict`), headers (authentication token, content type) and a body, usually **JSON**. The server replies with a **status code** (200 OK, 400 bad request, 422 invalid input, 500 server error, 503 unavailable) and a JSON body. An **API** is the contract: which URLs exist, what they accept, what they return.

A minimal model API in Python with FastAPI. Pydantic models define and validate the contract, so bad inputs are rejected before they reach the model:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
app = FastAPI(title="eta-model")

class EtaRequest(BaseModel):
    distance_km: float = Field(ge=0, le=50)
    prep_minutes: float = Field(ge=0, le=120)
    is_rush_hour: bool

class EtaResponse(BaseModel):
    eta_minutes: float
    model_version: str

# Load the model ONCE at startup, never per request.
W = np.array([2.1, 1.0, 6.0]); B = 4.0          # stand-in for a trained model
MODEL_VERSION = "eta-2026-11-01"

@app.get("/health")                    # used by the load balancer and Kubernetes
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=EtaResponse)
def predict(req: EtaRequest):
    x = np.array([req.distance_km, req.prep_minutes, float(req.is_rush_hour)])
    return EtaResponse(eta_minutes=round(float(W @ x + B), 1), model_version=MODEL_VERSION)
```

Run it locally with `uvicorn serve:app --port 8000` and open `http://localhost:8000/docs` for interactive documentation that FastAPI generates. Tested in-process with FastAPI's `TestClient`:

```python
from fastapi.testclient import TestClient
client = TestClient(app)
client.get("/health").json()          # {'status': 'ok'}
client.post("/predict", json={"distance_km": 3.2, "prep_minutes": 12, "is_rush_hour": True})
# 200 {'eta_minutes': 28.7, 'model_version': 'eta-2026-11-01'}
client.post("/predict", json={"distance_km": -1, "prep_minutes": 12, "is_rush_hour": True})
# 422 'Input should be greater than or equal to 0'
```

Three habits worth copying: return the **model version** with every prediction (debugging and A/B analysis need it); expose a **health endpoint**; validate inputs at the edge. Other protocols you will hear about: **gRPC** (binary, faster, used between internal services) and **WebSockets / server-sent events** (streaming, which is how LLM tokens arrive word by word).

## 42.4 Containers: "it works on my machine", solved

Your model needs a specific Python version, specific library versions, the model file and some system libraries. A **container image** packages all of that into one artefact; a **container** is a running instance of it. The image is built from a **Dockerfile**, a recipe:

```dockerfile
# Dockerfile (standard example, not run here)
FROM python:3.11-slim                      # base image: a small Linux with Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # pinned versions inside
COPY serve.py model.pt ./
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]


docker build -t eta-model:1.4.0 .              # build the image
docker run -p 8000:8000 eta-model:1.4.0        # run it, mapping port 8000
docker push registry.example.com/eta-model:1.4.0   # upload to a registry
```

Key ideas: images are **immutable** and **versioned by tag** (never deploy `latest` in production); they are stored in a **registry** (Docker Hub, ECR, GCR, Artifact Registry); a container is not a VM (it shares the host's kernel, so it starts in seconds and is light). For GPUs, base images from NVIDIA (CUDA) include the drivers' user-space libraries. Keep images small: large images slow down every deploy and scale-up.

## 42.5 Kubernetes: running many containers reliably

One container on one machine dies when the machine dies. **Kubernetes** (often "k8s") runs your containers across a **cluster** of machines and continuously makes reality match what you declared. The core objects:

- **Pod**: the smallest unit, one or more containers that run together. [Pods](https://kubernetes.io/docs/concepts/workloads/pods/).
- **Deployment**: "keep 6 copies of this pod running, version 1.4.0"; it replaces crashed pods and rolls out new versions gradually. [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/).
- **Service**: a stable network name and load balancing across the pods. [Services](https://kubernetes.io/docs/concepts/services-networking/service/).
- **Horizontal Pod Autoscaler (HPA)**: adds pods when CPU (or a custom metric such as requests per second) rises, removes them when it falls.
- **Probes**: *liveness* (restart me if this fails) and *readiness* (do not send me traffic until this passes, for example until the model has loaded).
- **Resource requests and limits**: how much CPU, memory and how many GPUs each pod needs; the scheduler places pods on machines with room.

```yaml
# deployment.yaml (standard example, not run here)
apiVersion: apps/v1
kind: Deployment
metadata: {name: eta-model}
spec:
  replicas: 6
  selector: {matchLabels: {app: eta-model}}
  template:
    metadata: {labels: {app: eta-model}}
    spec:
      containers:
      - name: api
        image: registry.example.com/eta-model:1.4.0
        ports: [{containerPort: 8000}]
        resources:
          requests: {cpu: "1", memory: "2Gi"}
          limits: {memory: "2Gi"}          # GPU pods add: nvidia.com/gpu: 1
        readinessProbe: {httpGet: {path: /health, port: 8000}}
        livenessProbe: {httpGet: {path: /health, port: 8000}}
```

You will rarely write these from scratch (platform teams provide templates, Helm charts or an internal deploy tool), but you should be able to read one and explain why a pod keeps restarting (failing liveness probe, out of memory: `OOMKilled`) or never receives traffic (readiness never passes, for example a slow model load). Managed alternatives that hide Kubernetes entirely: Cloud Run, AWS App Runner, SageMaker or Vertex AI endpoints, and serverless GPU platforms.

## 42.6 Storage: where data and models live

| Store | What it is for | Typical latency | Examples |
|---|---|---|---|
| App database (OLTP) | live product data: orders, users, menus; many small reads and writes | ~1–10 ms | PostgreSQL, MySQL, DynamoDB |
| Cache / key-value | features and results needed in milliseconds | <1 ms | Redis, Memcached |
| Data warehouse | analytics and training data; SQL over billions of rows | seconds | BigQuery, Snowflake, Redshift |
| Data lake / object storage | raw files, Parquet tables, model weights, checkpoints | tens of ms per file | S3, GCS, Azure Blob |
| Vector database | nearest-neighbour search over embeddings (Chapter 30) | ~1–50 ms | pgvector, FAISS-based services, Pinecone, Qdrant |

The rule: never query the warehouse inside an online request (too slow, too expensive); precompute and push what the online path needs into a cache or online feature store (Chapter 47). Model weights live in object storage and are downloaded when a container starts; a **model registry** (MLflow and similar) records which file is which version and which one is in production.

## 42.7 Queues and streams: decoupling systems

If the order service called the fraud model, the email service and the analytics pipeline directly, one slow consumer would slow every order. Instead it **publishes an event** ("order 123 placed") to a **queue** or **stream**, and each consumer reads at its own pace.

- **Queue** (SQS, RabbitMQ, Celery with Redis): each message is processed by one worker, then removed. Good for background jobs: "generate a description for this new menu item", "run the LLM classifier on this merchant".
- **Stream / log** (Kafka, Kinesis, Pub/Sub): an ordered, replayable log of events that many consumers can read independently. Good for real-time features ("orders in this area in the last 10 minutes"), streaming inference and feeding the warehouse.

Queues also absorb spikes: at dinner rush, messages wait a little longer instead of the system falling over. They make LLM pipelines robust too: put each item on a queue, have workers call the model with retries, and failed items go to a **dead-letter queue** for inspection instead of being lost.

## 42.8 Batch, online and streaming inference

| Mode | How | Latency | When to use | Delivery example |
|---|---|---|---|---|
| Batch | scheduled job scores everything, writes results to a table or cache | hours | predictions that do not depend on the moment; huge volumes; LLM jobs via batch APIs | nightly restaurant embeddings, weekly merchant categories |
| Online (real-time) | an API scores each request | ms | depends on context known only at request time | ranking the home feed, ETA at checkout |
| Streaming | a consumer scores each event as it arrives | seconds | react quickly to events, no user waiting | flag a suspicious order within seconds |
| Hybrid | batch precomputes the heavy part; online combines it cheaply | ms | most real systems | precomputed user and restaurant embeddings, online dot product + light reranker |

Scheduled jobs are run by an **orchestrator** (Airflow, Dagster, Prefect, or the cloud's own), which handles dependencies ("train only after the data job succeeds"), retries and alerts.

## 42.9 Latency, throughput and capacity

**Latency** is time per request; report percentiles, not averages, because users feel the slow tail. p99 = 99% of requests are faster than this. Measuring the toy API in-process:

```python
import time, numpy as np
lat = []
for _ in range(2000):
    t = time.perf_counter()
    client.post("/predict", json={"distance_km": 3.2, "prep_minutes": 12, "is_rush_hour": False})
    lat.append((time.perf_counter() - t) * 1000)
np.percentile(lat, [50, 95, 99])     # p50 2.78 ms, p95 3.15 ms, p99 3.60 ms (no network)
```

Real requests add network time, feature lookups and the model itself, and the tail grows under load.

**Throughput** is requests per second (RPS or QPS). **Little's law** links them: requests in flight = arrival rate × time each spends in the system. That sizes the number of copies you need:

```python
import numpy as np
def replicas_needed(peak_rps, latency_s, concurrency_per_replica, headroom=0.6):
    in_flight = peak_rps * latency_s                     # Little's law: L = lambda * W
    return int(np.ceil(in_flight / (concurrency_per_replica * headroom)))

replicas_needed(3000, 0.05, 8)    # 32 replicas for a 50 ms model at 3,000 RPS
replicas_needed(3000, 0.40, 8)    # 250 replicas if the model takes 400 ms
```

The 0.6 headroom keeps each replica at about 60% utilisation, because latency rises steeply as a server approaches 100% busy (queues build up). Notice the lesson: an 8× slower model needs about 8× the machines. Latency is a cost driver, not only a user-experience number.

Other terms you will hear: **SLO** (service-level objective, for example "p99 under 200 ms, 99.9% availability"), **timeouts** (never wait forever on a dependency), **fallbacks** (if the model is down, serve a popularity ranking), **cold start** (time for a new replica to load the model before it can serve).

## 42.10 Shipping changes safely: CI/CD and infrastructure as code

- **Version control (git)** for everything: code, configs, prompts, infrastructure.
- **CI (continuous integration)**: on every pull request, a service such as GitHub Actions or GitLab CI runs tests, linters and, for ML, a small evaluation.
- **CD (continuous delivery/deployment)**: after merge, build the image, push it to the registry, and deploy to staging, then production, often gradually (Chapter 47 covers shadow, canary and A/B releases).
- **Infrastructure as code**: Terraform (or Pulumi, CloudFormation) files describe buckets, databases, clusters and permissions, reviewed and versioned like code.
- **Secrets** (API keys, database passwords) never go in code or images; they come from a secret manager (AWS Secrets Manager, Vault) as environment variables at runtime.

## 42.11 Observability: seeing inside a running system

- **Logs**: timestamped records of events ("request 7f3 predicted eta=28.7 with model v1.4.0"). Structured (JSON) logs can be searched and aggregated. Redact personal data (Chapter 28).
- **Metrics**: numbers over time: request rate, error rate, latency percentiles, CPU, GPU memory, and ML metrics such as the mean prediction or the share of low-confidence outputs. Collected by Prometheus or a vendor, shown in Grafana or Datadog.
- **Traces**: the path of one request through every service, with the time spent in each; the fastest way to find where latency comes from. OpenTelemetry is the standard.
- **Alerts**: rules on metrics that page a person ("error rate above 2% for 5 minutes"), linked to a **runbook** saying what to do.

The four signals Google's SRE book recommends watching for any service: latency, traffic, errors and saturation. Chapter 47 adds the ML-specific layers (data drift, prediction drift, business metrics).

## 42.12 How it fits together: one request, end to end

A customer opens the app at 20:05 and the home feed is ranked:

1. The app sends `GET /feed?user=123&lat=…&lon=…` over HTTPS to the load balancer.
2. The load balancer forwards it to one of 20 pods of the ranking service (Kubernetes keeps 20 alive; the HPA added 8 for the dinner rush).
3. The service fetches the user's precomputed embedding and recent-order features from Redis (written last night by a batch job and updated by a stream consumer during the day).
4. It retrieves 500 candidate restaurants open nearby, calls the ranking model server, and gets scores in 30 ms.
5. It applies business rules (closed restaurants, delivery radius), logs the request, the scores and the model version to the event stream, and returns the top 50 in about 120 ms end to end.
6. Tonight, a scheduled job reads those logs from the warehouse, joins the orders that followed, retrains the model, evaluates it, registers it, and a CD pipeline rolls it out as a canary tomorrow.

If you can tell this story, with the reasons for each piece, you can hold the infrastructure part of any ML system design interview.

## Pitfalls

- Loading the model inside the request handler instead of once at startup.
- Querying the warehouse from an online request.
- Deploying the `latest` image tag, so you cannot tell what is running or roll back.
- Reporting average latency instead of p95/p99.
- Running servers near 100% utilisation and wondering why latency explodes.
- No timeout or fallback on a model call, so a slow model takes the whole page down.
- Secrets in code, notebooks or container images.
- Making everything real-time when a nightly batch job would do.

## Interview questions

**Beginner / mid level**

1. What is the difference between a container and a virtual machine?
2. What do a Kubernetes Deployment, Service and HPA each do?
3. When would you use batch inference instead of an online API?
4. What do p50 and p99 latency mean, and why report both?
5. Why put a queue between two services?

**Senior level**

6. Your model API meets its p50 target but p99 is five times higher. What could cause it and what would you check?
7. Estimate how many replicas you need for 2,000 RPS with 80 ms latency and 4 concurrent requests per replica.
8. Design the infrastructure for scoring 10 million menu items with an LLM every week, robust to API failures.
9. A new model version doubles CPU per request. What changes operationally, and how would you roll it out?

Short answers: (1) A VM includes a full operating system on virtualised hardware; a container shares the host kernel and only packages the app and its dependencies, so it is smaller and starts in seconds. (3) When predictions do not depend on request-time context, volume is large, or cost matters more than freshness. (6) Garbage-collection pauses, cold replicas, slow feature lookups, a few huge inputs, CPU saturation or noisy neighbours; check traces, per-replica metrics and the input size distribution. (7) In flight = 2000 × 0.08 = 160; at 4 per replica and 60% headroom, 160 / 2.4 ≈ 67 replicas. (8) Items on a queue or in batch files; workers or the provider's batch API; retries with backoff; dead-letter queue; idempotent writes keyed by item id and model version; skip items whose text has not changed; monitor cost and failure rate. (9) Twice the replicas or bigger machines for the same traffic; update resource requests and autoscaling limits; load-test; canary release with latency and error guardrails.

## Practice

- Run the FastAPI example locally, call it with `curl`, then package it in a Docker image and run the container.
- Deploy the container to a free or cheap managed platform (Cloud Run, Render, Hugging Face Spaces) and measure p50/p95 latency from your laptop.
- Install `kind` or `minikube`, deploy the YAML above, delete a pod and watch Kubernetes recreate it.
- Draw the two-path diagram for a system you know (search, fraud, recommendations) and label which parts are batch, online and streaming.
- Redo the capacity arithmetic for three latency targets and explain the cost difference.

## Go deeper

- [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/) (MIT): shell, git, debugging; the fastest way to get comfortable on servers.
- [Docker: get started](https://docs.docker.com/get-started/) and [What is a container?](https://www.docker.com/resources/what-container/)
- [Kubernetes concepts overview](https://kubernetes.io/docs/concepts/overview/), then [Pods](https://kubernetes.io/docs/concepts/workloads/pods/), [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/), [Services](https://kubernetes.io/docs/concepts/services-networking/service/) and [autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/).
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/): building and testing APIs in Python.
- [MDN: An overview of HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview).
- [Google SRE book](https://sre.google/sre-book/table-of-contents/), especially the chapters on monitoring and service-level objectives.
- [What is Terraform](https://developer.hashicorp.com/terraform/intro) and [What is OpenTelemetry?](https://opentelemetry.io/docs/what-is-opentelemetry/)
- Chip Huyen, [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) (O'Reilly, 2022): chapters on deployment and infrastructure, written for data scientists.
- [Made With ML](https://madewithml.com/) (Goku Mohandas): a free course that takes one ML project from notebook to production.

## Done when

You can draw the online and offline paths of an ML system and name each component. You can explain containers, images, registries, pods, deployments, services and autoscaling in plain words. You have built and run a model API in a Docker container yourself.

You can choose batch, online, streaming or hybrid inference for a use case and justify it. You can size replicas with Little's law and explain why latency drives cost. You know what logs, metrics, traces and alerts are each for.

---

[← 41 · NLP tasks with modern tools](41-nlp-tasks-with-modern-tools.md) · [Contents](../README.md#contents) · [43 · Recommender systems →](43-recommender-systems.md)
