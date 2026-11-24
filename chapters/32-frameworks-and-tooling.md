# 32 · Frameworks and tooling

Part IX · Retrieval and agents. Assumes Chapters 22 (prompting), 31 (RAG) and 33 (agents) at least in outline: this chapter is about the tools used to build them.

## Why it matters

Job ads list LangChain, LangGraph, LlamaIndex and Hugging Face, and interviewers ask about them, but what they really test is judgement: when a framework saves time, when it hides bugs, and whether you could build the same thing in plain Python. Framework APIs change every few months, so this chapter teaches the concepts and shapes, not exact signatures. **All framework snippets here are untested sketches: check the current documentation before using them.** The plain-Python code was run.

The chapter starts with a one-table map of the landscape (32.1), then covers the frameworks you are most likely to meet: LangChain (32.2) and LangGraph (32.3) for chains and agents, LlamaIndex for the data side of RAG (32.4), DSPy for optimising prompts as parameters (32.5), and the Hugging Face stack for models (32.6). Section 32.7 gives a decision rule for whether to use a framework at all, 32.8 describes the different stack research scientists use, and 32.9 shows a RAG step written in plain Python.

Frameworks change quickly; the abstractions underneath them do not. A chain, a state machine, a retriever and an optimisable prompt are the same ideas in every library. Learning the ideas lets you pick up any framework in an afternoon, and recognise when the framework is adding complexity rather than removing it.

## Key concepts

- **LangChain and LCEL**: a library of model wrappers, prompts, retrievers and parsers, composed into chains. [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview).
- **LangGraph**: stateful graphs of steps with loops, branches, persistence and human approval steps. [LangGraph documentation](https://langchain-ai.github.io/langgraph/).
- **LlamaIndex**: loaders, chunking, indexes and query engines for RAG. [LlamaIndex documentation](https://developers.llamaindex.ai/python/framework/).
- **DSPy**: declare what each step does and let an optimiser tune prompts against a metric. [dspy.ai](https://dspy.ai/).
- **Hugging Face stack**: models, datasets, PEFT and TRL. [PEFT](https://huggingface.co/docs/peft/index), [TRL](https://huggingface.co/docs/trl/index), [LLM course](https://huggingface.co/learn/llm-course/chapter1/1).
- **Agent SDKs and MCP**: thin, provider-maintained agent loops (OpenAI Agents SDK, Claude Agent SDK, Google ADK) and the Model Context Protocol for exposing tools to any of them (Chapter 33). [modelcontextprotocol.io](https://modelcontextprotocol.io/).
- **Constrained decoding**: libraries that force output to match a JSON schema or grammar at sampling time (Outlines, XGrammar, the structured-output modes of vLLM and SGLang; Chapter 21).
- **Framework vs plain Python**: frameworks speed up prototypes; simple production flows are often clearer with a provider SDK and your own functions. [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) makes the case for simple patterns.

## 32.1 The landscape in one table

| Tool | What it is for | Core abstraction |
|---|---|---|
| LangChain | Glue for LLM apps: model wrappers, prompts, retrievers, tools, output parsers | Runnables composed with LCEL (`prompt \| model \|` `parser`) |
| LangGraph | Stateful, multi-step agents and workflows with loops, branching, human-in-the-loop, persistence | A graph of nodes that read and update a typed state |
| LlamaIndex | Data ingestion and retrieval for RAG: loaders, chunking, indexes, query engines | Documents → nodes → index → query engine |
| DSPy | Programming (not prompting) LLM pipelines; optimises prompts and few-shot examples against a metric | Signatures and modules, compiled by optimisers |
| Hugging Face | Models, datasets, tokenizers, training, PEFT, serving | `transformers`, `datasets`, `peft`, `trl`, `accelerate`, Hub |
| Provider SDKs | Direct calls to OpenAI, Anthropic, Google, etc. | Messages, tools, structured outputs |
| Instructor / Pydantic AI | Typed structured outputs and light agents | Pydantic models as the contract |
| Agent SDKs (OpenAI Agents SDK, Claude Agent SDK, Google ADK) | A provider's own agent loop: tool calling, hand-offs, sub-agents, guardrails, tracing | An agent with tools, run in a loop until done |
| MCP (Model Context Protocol) | A standard way to expose tools, resources and prompts to any compatible application | Servers that publish tools; clients inside the app that call them |
| Constrained decoding (Outlines, XGrammar, vLLM/SGLang structured outputs) | Output guaranteed to parse: JSON schemas, regexes, grammars | A token mask applied at every sampling step |
| Observability (LangSmith, Langfuse, Phoenix, MLflow tracing) | Traces, prompt versions, evaluations | Spans per LLM or tool call |

## 32.2 LangChain and LCEL

LangChain's useful core is small: a common interface (`invoke`, `batch`, `stream`, async variants) for models, prompts, retrievers and parsers, and the pipe operator to compose them.

```python
# Sketch, check current docs. Packages are split: langchain-core, langchain-openai, ...
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the context. If it is not there, say you don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
chain.invoke("Can I change the delivery address after ordering?")
chain.batch(questions, config={"max_concurrency": 8})   # parallel calls for free
```

**Where LangChain is now.** LangChain 1.0 (late 2025) moved its centre of gravity from LCEL chains to agents: a `create_agent` entry point built on LangGraph's runtime, with middleware hooks for things such as summarising long histories, human approval and guardrails. LCEL still works and is still the clearest way to understand the library's composition model, which is why it is shown here, but new agent code is usually written against the agent API. Check the current documentation before an interview that names LangChain, since the API has changed more than once.

What it gives you: swapping providers by changing one object, batching and streaming without writing them, many integrations (vector stores, loaders), and tracing when paired with LangSmith. What it costs: deep abstraction layers that make errors hard to read, frequent breaking changes, and prompts hidden inside library defaults (always print the final prompt). Tools are declared with a decorator and bound to a model (`llm.bind_tools([...])`); structured output with `llm.with_structured_output(PydanticModel)`.

## 32.3 LangGraph

LangGraph models an agent or workflow as a state machine: nodes are functions that take the state and return updates; edges (fixed or conditional) decide what runs next; loops are allowed. On top it adds checkpointing (resume a run, time-travel debugging), human-in-the-loop interrupts (pause before a refund is issued), streaming of intermediate steps, and multi-agent patterns.

The idea fits in 30 lines of plain Python, which is worth writing once so the framework holds no mystery:

```python
from typing import TypedDict, Callable

class State(TypedDict, total=False):
    query: str; intent: str; docs: list; answer: str; attempts: int
def classify(s): return {**s, "intent": "refund" if "refund" in s["query"].lower() else "faq"}
def retrieve(s):
    kb = {"refund": ["Refunds go to the original payment method within 5-7 days."],
          "faq": ["Delivery fees depend on distance."]}
    return {**s, "docs": kb[s["intent"]]}
def generate(s): return {**s, "answer": f"[{s['intent']}] " + s["docs"][0],
                        "attempts": s.get("attempts", 0) + 1}
def check(s): return "end" if s["answer"] and s["attempts"] >= 1 else "generate"

class Graph:
    def __init__(self): self.nodes, self.edges, self.cond = {}, {}, {}
    def add_node(self, name, fn): self.nodes[name] = fn
    def add_edge(self, a, b): self.edges[a] = b
    def add_conditional(self, a, router: Callable): self.cond[a] = router
    def run(self, start, state, max_steps=20):
        node, trace = start, []
        for _ in range(max_steps):                # always cap the loop
            state = self.nodes[node](state); trace.append(node)
            nxt = self.cond[node](state) if node in self.cond else self.edges.get(node, "end")
            if nxt == "end":
                return state, trace
            node = nxt
        raise RuntimeError("step limit hit")

g = Graph()
for name, fn in [("classify", classify), ("retrieve", retrieve), ("generate", generate)]:
    g.add_node(name, fn)
g.add_edge("classify", "retrieve"); g.add_edge("retrieve", "generate");
g.add_conditional("generate", check)
state, trace = g.run("classify", {"query": "I want a refund for order 123"})
# trace -> ['classify', 'retrieve', 'generate']
# answer -> "[refund] Refunds go to the original payment method within 5-7 days."
```

The LangGraph version has the same shape (sketch, check docs):

```python
from langgraph.graph import StateGraph, START, END
builder = StateGraph(State)
builder.add_node("classify", classify); builder.add_node("retrieve", retrieve); builder.add_node("generate", generate)
builder.add_edge(START, "classify"); builder.add_edge("classify", "retrieve"); builder.add_edge("retrieve", "generate")
builder.add_conditional_edges("generate", check, {"generate": "generate", "end": END})
graph = builder.compile(checkpointer=memory_saver)      # persistence + interrupts
graph.invoke({"query": "..."}, config={"configurable": {"thread_id": "order-123"}})
```

Use it when the flow has loops, branches, long-running state or approval steps (a support agent that can issue vouchers). For a fixed three-step pipeline it is overkill. Chapter 33 covers agent design itself.

## 32.4 LlamaIndex

Strongest at the data side of RAG: dozens of loaders (PDF, Notion, SQL), node parsers for chunking, index types (vector, keyword, summary, knowledge graph), retrievers with metadata filters, rerankers, and query engines that combine them (sub-question decomposition, router across indexes). A quick prototype is a few lines:

```python
# Sketch, check current docs
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
docs = SimpleDirectoryReader("policies/").load_data()
index = VectorStoreIndex.from_documents(docs)
answer = index.as_query_engine(similarity_top_k=5).query("What is the refund window for groceries?")
```

The defaults (chunk size, embedding model, prompt) are rarely right for production; you will end up configuring each piece, at which point the value is the integrations and the structure, not the magic.

## 32.5 DSPy

DSPy treats prompts as parameters to be optimised. You declare *what* a step does with a signature, compose modules, give a metric and some examples, and an optimiser searches over instructions and few-shot demonstrations.

```python
# Sketch, check current docs
import dspy
class TagDish(dspy.Signature):
    """Assign cuisine and dietary tags to a menu item."""
    item_name: str = dspy.InputField()
    description: str = dspy.InputField()
    cuisine: str = dspy.OutputField()
    is_vegetarian: bool = dspy.OutputField()

tagger = dspy.ChainOfThought(TagDish)
optimiser = dspy.MIPROv2(metric=tag_accuracy, auto="light")
tuned = optimiser.compile(tagger, trainset=labelled_items)
```

Good fit: a pipeline with a clear metric and a few hundred labelled examples, where you want to switch models without rewriting prompts by hand. Menu-item tagging against a fixed food taxonomy (the applied case study used in Part XI) is a good example: the taxonomy gives you the label set and the metric. Weak fit: open-ended chat without a metric.

### What the optimiser actually does

"An optimiser searches over prompts" is the sentence that loses points in an interview, because it hides the three steps and their cost. MIPROv2, the optimiser in the snippet above, works as follows:

1. **Bootstrap demonstrations.** Run the program on training examples, usually with a strong "teacher" model, and keep the full traces (inputs, intermediate outputs, final answer) of the runs that pass the metric. Each kept trace becomes a candidate few-shot demonstration for every module it passed through, so the demonstrations include intermediate steps you never labelled.
2. **Propose instructions.** Show an LLM a summary of the dataset, the program's code and some of the bootstrapped demonstrations, and ask it for several candidate instructions per module.
3. **Search the combinations.** Treat each module's (instruction, demonstration set) pair as a discrete hyperparameter and run Bayesian optimisation over the combinations: each trial evaluates one combination on a minibatch of the validation set, a surrogate model proposes the next combination to try, and the best candidates are re-scored on the full validation set.

**GEPA** (2025) replaces the black-box search with reflection: an LLM reads execution traces and the metric's feedback (ideally textual, such as "the citation was missing"), proposes targeted edits to the instructions, and keeps a Pareto front of candidates that are best on different examples. Its authors report better results than MIPROv2 with fewer rollouts; treat that as a claim to check on your own task.

**What a compile costs.** Do the arithmetic before you run it. As an illustration, a two-module program with 25 search trials on minibatches of 35 examples makes 25 × 35 × 2 = 1,750 calls; bootstrapping on 100 examples adds 200, instruction proposals about 20, and a final pass over 300 validation examples 600, about 2,570 calls in total. At 1,500 tokens a call and $3 per million input tokens that is about $12: cheap compared with an engineer's afternoon, but not free if you recompile on every change.

**Where it goes wrong.** The validation set is the training signal, so it needs to be large enough to rank candidates. At 80% accuracy, a 35-example minibatch has a standard error of √(0.8 × 0.2 / 35) = 6.8 points, and even 300 examples give 2.3 points, so the winning prompt is partly the luckiest one: the winner's curse of Chapter 34, applied to prompts. Hold out a separate test set, compare the compiled program with a hand-written prompt on it, and recompile when you change the model, since the selected demonstrations are tuned to the model that produced them.

## 32.6 The Hugging Face stack

- `transformers`: models and tokenizers, `pipeline()` for quick inference, `Trainer` for fine-tuning (Chapter 41).
- `datasets`: memory-mapped datasets with fast `map` / `filter`, streaming for large corpora.
- `peft`: LoRA, QLoRA, prefix and prompt tuning (Chapter 20).
- `trl`: SFT, DPO, KTO, GRPO trainers (Chapters 18 and 19).
- `accelerate`: the same script on CPU, one GPU or many.
- `sentence-transformers`: embedding and cross-encoder models (Chapter 29).
- `text-generation-inference`, `text-embeddings-inference`: serving. For new LLM-serving deployments, vLLM and SGLang are now the usual choice (Chapter 21); check TGI's current maintenance status before adopting it. TEI remains a common choice for embedding models.
- The Hub: model cards, licences, versions. Pin a model by revision hash in production.

## 32.7 Framework or plain Python?

A decision rule worth saying in interviews:

- **Prototype or hackathon**: use the framework; speed matters more than control.
- **Production with a simple flow** (one retrieval, one call, structured output): provider SDK + Pydantic + your own small functions. Fewer dependencies, readable stack traces, no prompt surprises.
- **Production with complex control flow** (loops, approvals, long-running state): LangGraph or a workflow engine is worth it, but keep business logic in your own functions so you can leave the framework later.
- **Always**: own your prompts as versioned files, log every call (inputs, outputs, tokens, latency), and keep an evaluation set that runs in CI. These matter more than the framework choice.

Team fit counts too: if the platform team already supports one stack with tracing and deployment, the cost of a different choice is not just yours.

## 32.8 The research stack

Application frameworks are not what research scientists use to train models. The tools you will meet in a research team:

| Need | Common tools |
|---|---|
| Pretraining at scale | Megatron-LM / Megatron Core, torchtitan, DeepSpeed, nanotron, MaxText (JAX) |
| Fine-tuning and post-training | Hugging Face TRL, PEFT, Axolotl, torchtune; OpenRLHF, verl and similar for large-scale RL |
| Inference for rollouts and evaluation | vLLM, SGLang |
| Data processing | datatrove, Dolma toolkit, Spark or Ray for large pipelines; NeMo Curator |
| Evaluation | lm-evaluation-harness, Inspect, HELM, LightEval, OpenAI's simple-evals style scripts |
| Interpretability | TransformerLens, nnsight, SAELens |
| Experiment tracking | Weights & Biases, MLflow, TensorBoard |
| Cluster scheduling | Slurm, Kubernetes with Kueue or similar |
| Kernels | Triton, CUDA, FlashAttention, Liger-style fused kernels |

The habits from Chapter 34 matter more than the tools: configuration files, versioned code and data, tracked runs and evaluation as code. Know one stack deeply and be able to read the others' code.

### Choosing within the research stack

The table is a list; interviewers want the judgement behind it. Four choices come up most often.

**FSDP2 and torchtitan, or Megatron.** Fully sharded data parallelism (ZeRO-3 style, Chapter 17) shards parameters, gradients and optimiser state across data-parallel ranks and gathers each layer just before it is used. It needs no change to the model code and is enough while each layer fits comfortably on a GPU and the global batch is large enough to give every rank useful work. Once you need tensor parallelism (layers too large, or more GPUs than the batch can feed), pipeline parallelism or expert parallelism for a large MoE, Megatron-Core has the most mature implementations and kernels. torchtitan sits between: it composes FSDP2 with tensor, pipeline and context parallelism in native PyTorch, and is easier to read and modify. A reasonable default is torchtitan or FSDP2 for research code you will change weekly, and Megatron-Core (or the lab's in-house stack) for a large production pretraining run.

**JAX or PyTorch.** JAX expresses parallelism by annotating how arrays are sharded and letting the compiler insert the communication, which makes large-scale sharding concise and is the natural choice on TPUs (MaxText). The costs are a functional programming style, compile times, and debugging through traced code. PyTorch is eager by default, dominant on GPUs and in open-source research code. Pick by hardware first, then by the team's existing code.

**verl or OpenRLHF, and colocated or disaggregated rollouts.** RL for LLMs alternates generation (an inference engine such as vLLM) with training (Chapter 19). A **colocated** design runs both on the same GPUs and switches between them, resharding weights each time; nothing sits idle waiting for the other phase, which suits synchronous on-policy RL. A **disaggregated** design puts generation and training on separate GPU pools and streams samples between them; it allows asynchronous RL and different parallelism for each side, at the cost of idle capacity when the two are unbalanced and of slightly stale samples. verl's design centres on colocation, while OpenRLHF began with a Ray-based layout that places each model on its own GPUs; both have since added the other mode, so ask which mode a team runs rather than which library.

**lm-evaluation-harness or Inspect.** The harness is built for standard benchmarks: log-likelihood scoring of multiple-choice answers, few-shot templates, and numbers comparable with published papers. Inspect is built for evaluations with tools, multi-turn interaction, sandboxes and model-graded scoring, which is why safety and agent evaluations increasingly use it (Chapters 23 and 25). Many teams use both: the harness for pretraining checkpoints, Inspect for post-trained models and agents.

## 32.9 A worked example: one RAG step in plain Python

Here is the core of a production RAG step without a framework, to show what a framework would otherwise hide:

- a versioned prompt;
- retries with backoff on transient errors;
- a response cache keyed by the model, prompt version, temperature and exact prompt, written only after the output passes every check;
- schema validation with Pydantic;
- a citation check against the retrieved ids;
- a trace of every step.

```python
import hashlib, json, random, time
from pydantic import BaseModel, ValidationError

class Answer(BaseModel):
    answer: str
    citations: list[str]

class TransientError(Exception):
    pass

PROMPT_VERSION = "rag-answer-v3"
PROMPT_V3 = ("Answer from the passages only. Return JSON with 'answer' and 'citations' (passage ids).\n"
             "If the passages do not answer the question, say so.\n\nPassages:\n{passages}\n\nQuestion: {question}")

def call_with_retries(llm, prompt, attempts=3, base_delay=0.5):
    for i in range(attempts):
        try:
            return llm(prompt)
        except TransientError:                       # rate limits, timeouts: retry with jittered backoff
            if i == attempts - 1:
                raise
            time.sleep(base_delay * 2 ** i * (1 + random.random()))

def answer(question, retrieve, llm, cache, trace, model="small-model", temperature=0.0, k=5):
    passages = retrieve(question, k=k)
    trace.append({"step": "retrieve", "ids": [p["id"] for p in passages]})
    prompt = PROMPT_V3.format(passages="\n".join(f"[{p['id']}] {p['text']}" for p in passages), question=question)
    key = hashlib.sha256(json.dumps([model, PROMPT_VERSION, temperature, prompt]).encode()).hexdigest()
    raw = cache.get(key) if temperature == 0 else None   # never serve a sampled output from cache
    cached = raw is not None
    if not cached:
        t0 = time.time(); raw = call_with_retries(llm, prompt)
        trace.append({"step": "llm", "cached": False, "seconds": round(time.time() - t0, 3)})
    else:
        trace.append({"step": "llm", "cached": True})
    try:
        out = Answer.model_validate_json(raw)
    except ValidationError:
        trace.append({"step": "parse", "ok": False}); return None
    unknown = set(out.citations) - {p["id"] for p in passages}
    trace.append({"step": "check", "unknown_citations": sorted(unknown)})
    if unknown:
        return None
    if not cached and temperature == 0:
        cache[key] = raw                                  # cache only outputs that passed every check
    return out
```

The check script runs it with a fake model that fails once with a rate-limit error. The first call retries and succeeds. The second, identical call is served from the cache. The trace reads retrieve, llm, check, retrieve, llm (cached), check.

**Two cache rules that are easy to get wrong.** First, cache only outputs that passed validation and the citation check. A cache written before validation stores a malformed or hallucinated answer and then serves it to every later user who asks the same question, with no model call that could have fixed it. In the check script, a response citing a passage that was not retrieved is rejected twice, reaches the model twice, and leaves the cache empty. Second, put everything that changes the output into the key (the model, the prompt version, the temperature) and do not serve sampled outputs (temperature > 0) from the cache at all: caching one sample hides the variance you would otherwise see, and makes an evaluation run on cached answers look more stable than the product is. A cache hit also skips the latency and cost you are trying to measure, so evaluation runs should bypass it.

The whole thing is about 40 lines, every line is visible in a stack trace, and each piece maps to a framework feature:

- the retry wrapper corresponds to a framework's retry policy;
- the cache to its LLM cache;
- the Pydantic model to its output parser;
- the trace list to its callbacks or tracing integration.

For a flow like this, plain code is easier to test and to change. When the flow grows loops, branches, human approvals and state that must survive a restart, the bookkeeping is what LangGraph or a workflow engine provides (32.3, 32.7), and at that point the framework earns its keep. Either way, keep functions like `answer` as plain, testable code that the framework calls.

## Pitfalls

- Not knowing what prompt the framework actually sends. Log it.
- Pinning nothing: a minor version bump changes behaviour in production.
- Copying a tutorial's chunk size, embedding model and top-k into production without evaluation.
- Agent loops without a step limit or cost cap.
- Hard-wiring business rules inside framework callbacks, making migration painful.
- Choosing a framework because it is in the job ad rather than because it fits the problem.

## Interview questions

**Mid level**

**1. What does LangChain's LCEL give you over calling the API directly?** A common interface (`invoke`, `batch`, `stream` and async versions) for every component, and the pipe operator to compose prompts, models, retrievers and parsers into one runnable, so batching, streaming, retries and tracing come with the composition rather than being written by hand. The concrete win is concurrency: 1,000 calls of 2 seconds each take 2,000 s in a plain loop and about 250 s with `batch(..., max_concurrency=8)`. The cost is indirection: errors surface deep in library frames and the final prompt is assembled out of sight, so log it. Follow-up: "would you use it in production?" For a simple flow, an SDK call inside a thread pool gives the same concurrency with fewer dependencies.

**2. What is the difference between LangChain and LangGraph?** LangChain supplies components and linear composition; LangGraph models a workflow as a state machine whose nodes read and update a typed state, with conditional edges, loops, checkpointing after each step, human-in-the-loop interrupts and streaming of intermediate steps. Use LangChain-style composition for a fixed pipeline and LangGraph when the flow loops (retry until a check passes), branches, or must pause and resume. Since LangChain 1.0 its agents run on LangGraph's runtime, so the line is now components versus orchestration. Follow-up: "what does the checkpointer buy you?" Resuming after a crash or an approval pause, and replaying a run from any step for debugging.

**3. When would you reach for LlamaIndex?** When the hard part is the data side of RAG: many sources to load (PDFs, wikis, databases), chunking and metadata, several index types, and query engines that route or decompose questions across them. It gets a prototype running in a few lines, but its defaults (chunk size, embedding model, top-k, prompt) are rarely right, and every one of them moves cost and quality: 5 chunks of 512 tokens is 2,560 context tokens a query, which at 100,000 queries a day and $3 per million tokens is about $770 a day. When not to: a single, well-structured corpus that one retriever and one prompt already handle. Follow-up: "what do you tune first?" Chunking and retrieval, measured with recall@k on a labelled set (Chapters 25 and 31).

**4. What does DSPy optimise?** The prompt of each module in a pipeline: its instruction text and its few-shot demonstrations, chosen to maximise a metric on labelled examples. MIPROv2 bootstraps demonstrations from successful traces, has an LLM propose instructions, and searches the combinations with Bayesian optimisation on validation minibatches; GEPA instead reflects on traces and edits instructions. A two-module compile is a few thousand LLM calls (about $12 in 32.5's example), and with 300 validation examples at 80% accuracy the noise is ±2.3 points of standard error, so hold out a test set. It does not change model weights. Follow-up: "when is it not worth it?" When you have no metric, or fewer than a few dozen labelled examples.

**Senior level**

**5. Your team's RAG service is built on LangChain and debugging is painful. Would you migrate? How would you decide and how would you do it safely?** Measure first: list the incidents and the hours lost to the framework over a quarter, and the features you actually use; if it is three wrappers and a retriever, migration is cheap, and if you rely on many integrations, it is not. Migrate incrementally behind the same interface: rewrite one step at a time as plain functions, run old and new side by side on the evaluation set and in shadow mode on live traffic, compare traces, then switch. Size the comparison: on 500 evaluation questions at 90% correctness, the difference between two independent runs has a standard error of 1.9 points (±3.7 at 95%), and scoring both versions on the same questions shrinks it. When not to migrate: if the pain is missing tracing, add tracing first. Follow-up: "how long?" Days for a simple flow; the shadow period, not the code, sets the timeline.

**6. Design a refund-handling agent with a human approval step. Which tool would you use and how does state persist across the pause?** A LangGraph-style state machine: classify, gather evidence with read-only tools, propose a refund, check it against policy in code, and interrupt before the payment node when the amount or risk is above a threshold. A checkpointer keyed by conversation id stores the state after every step in a durable database, so the run survives restarts and resumes when the approver's decision arrives, with a timeout that escalates stale cases and an audit log of every step. Scale it: at 10,000 refunds a day with 5% needing approval, about 500 paused threads exist at once if approvals take a day, so the checkpoint store, not the model, is the component to size. Follow-up: "what if the order changed while paused?" Re-read the order before the payment node runs; never trust state captured before the pause.

**7. How do you version, test and roll out a prompt change in a framework-based app?** Keep prompts as versioned files, not strings inside library calls; run the evaluation set in CI with thresholds that block a merge; tag every trace with the prompt version; roll out as a canary and keep a one-line rollback (Chapter 47). Be honest about what the canary can detect: to see a resolution rate fall from 80% to 75% with 80% power needs about 1,090 conversations per arm, which is 11 days at 100 canary conversations a day, so the canary guards against crashes and gross failures while the offline evaluation guards quality. Follow-up: "what if the provider updates the model under you?" Pin the model version where the API allows, and rerun the evaluation set on a schedule to catch silent drift.

**8. How would you make framework choice a team decision rather than a personal preference?** Write a short RFC with explicit criteria: the flows the team must support, debuggability, tracing and evaluation integration, deployment fit, licence and dependency risk, and how hard it is to leave. Run a time-boxed spike that builds the same representative flow (for example, RAG with one tool call and an approval) two or three ways and measures lines of code, latency, and time to find a planted bug. Include the platform team's cost: supporting a second stack in tracing and deployment costs real time, and if it takes one engineer-week a month, that is about 23% of an engineer. Decide, record the decision and its review date. Follow-up: "what if a senior engineer disagrees?" Ask which criterion they weigh differently and test that one in the spike.

**9. Constrained decoding or validate-and-retry for structured output?** Constrained decoding masks tokens that would break a JSON schema or grammar at each sampling step, so the output always parses; validate-and-retry samples freely and re-asks on failure. Retrying is cheap when failures are rare: at a 3% failure rate the expected number of calls is 1/0.97 = 1.03. Constrained decoding guarantees syntax, not meaning, and a schema that forces the answer field first stops the model reasoning before it commits, so put a reasoning field first or keep the schema loose. When not to constrain: free-text tasks where the schema is only a wrapper. Follow-up: "where is it implemented?" As a logit mask in the serving engine (vLLM, SGLang, XGrammar, Outlines) or through a provider's structured-output mode (Chapter 21).

## Practice

1. Build the same small RAG over 50 policy documents three ways: plain Python + SDK, LangChain, LlamaIndex. Compare lines of code, latency, and how long it takes to find a planted bug.
2. Re-implement the plain-Python graph with LangGraph and add a human-approval interrupt.
3. Use DSPy to optimise a classifier or tagger on 200 labelled items and compare with your hand-written prompt.
4. Add the two cache rules of 32.9 to your own RAG code, then write a test that a response with an invented citation is not cached.
5. **Research exercise.** Measure how much prompt optimisation results depend on noise. Compile the same DSPy program three times with different random seeds (and, if you can afford it, two validation-set sizes, 50 and 300), then score every compiled program and your hand-written prompt on one fixed, held-out test set of at least 300 items.

<details><summary>Hint and expected result</summary>

Item 1: the plain-Python version is usually the shortest for a single retrieval and call (the 32.9 core is about 40 lines) and the fastest to debug; the frameworks win on loaders and integrations. Item 2: the interrupt should pause before the approval node, and resuming with the same `thread_id` should continue from the saved state, not restart. Item 3: expect a gain of a few points over a careful hand-written prompt on a held-out set, and more over a naive one; if the gain is smaller than about 2 standard errors of your test set (2.3 points at 300 items and 80% accuracy), call it no difference. Item 4: the test should check both that `answer` returns `None` and that the cache stays empty. Item 5: expect the three seeds to disagree on which instructions and demonstrations win, and the spread of test scores across seeds to be comparable with the gain over the hand-written prompt when the validation set is small; with 300 validation items the spread should shrink. Report the mean and spread, not the best compile.

</details>

## Read next

- LangChain documentation: LCEL and "How-to" guides; LangGraph documentation: concepts (state, checkpoints, interrupts).
- LlamaIndex documentation: "Building RAG from scratch" and the production RAG guides.
- Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023).
- Anthropic, "Building effective agents" (blog post, 2024): a strong argument for simple patterns over frameworks.
- Hugging Face course (free), chapters on Transformers, Datasets and fine-tuning.
- Links to the documentation of each tool are in the Key concepts section at the top of this chapter.

## Done when

- In one sentence each, without notes, you state what LangChain, LangGraph, LlamaIndex, DSPy, the Hugging Face stack, the agent SDKs, MCP and constrained-decoding libraries are for.
- From a blank file, you write the plain-Python state-graph runner of 32.3, with a step cap, in 15 minutes, and list three things LangGraph adds (checkpointing, interrupts, streaming).
- You explain MIPROv2's three steps in two minutes and estimate a compile's cost: two modules, 25 trials, minibatches of 35, about 2,570 calls.
- You find both cache bugs in a version of 32.9 that caches before validation and ignores temperature, and fix them in 10 minutes.
- For a given research setting, you choose FSDP2 or Megatron, and colocated or disaggregated RL rollouts, and justify each choice in two sentences.
- You compute, in under two minutes, how many canary conversations are needed to detect a drop from 80% to 75% (about 1,090 per arm).
- You have one project where you built the same thing with and without a framework, with the line counts and the debugging times written down.

---

[← 31 · Retrieval-augmented generation](31-retrieval-augmented-generation.md) · [Contents](../README.md#contents) · [33 · Agents and tool use →](33-agents-and-tool-use.md)
