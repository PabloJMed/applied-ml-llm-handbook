# 20 · Frameworks and tooling

## Why it matters

Job ads list LangChain, LangGraph, LlamaIndex and Hugging Face, and interviewers ask about them, but what they really test is judgement: when a framework saves time, when it hides bugs, and whether you could build the same thing in plain Python. Framework APIs change every few months, so this chapter teaches the concepts and shapes, not exact signatures. **All framework snippets here are untested sketches: check the currentdocumentation before using them.** The plain-Python code was run.

## Key concepts

- **LangChain and LCEL**: a library of model wrappers, prompts, retrievers and parsers, composed into chains. [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview).
- **LangGraph**: stateful graphs of steps with loops, branches, persistence and human approval steps. [LangGraph documentation](https://langchain-ai.github.io/langgraph/).
- **LlamaIndex**: loaders, chunking, indexes and query engines for RAG. [LlamaIndex documentation](https://developers.llamaindex.ai/python/framework/).
- **DSPy**: declare what each step does and let an optimiser tune prompts against a metric. [dspy.ai](https://dspy.ai/).
- **Hugging Face stack**: models, datasets, PEFT and TRL. [PEFT](https://huggingface.co/docs/peft/index), [TRL](https://huggingface.co/docs/trl/index), [LLM course](https://huggingface.co/learn/llm-course/chapter1/1).
- **Framework vs plain Python**: frameworks speed up prototypes; simple production flows are often clearer with a provider SDK and your own functions. [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) makes the case for simple patterns.

## 20.1 The landscape in one table

| Tool | What it is for | Core abstraction |
|---|---|---|
| LangChain | Glue for LLM apps: model wrappers, prompts, retrievers, tools, output parsers | Runnables composed with LCEL (`prompt \| model \|` `parser`) |
| LangGraph | Stateful, multi-step agents and workflows with loops, branching, human-in-the- loop, persistence | A graph of nodes that read and update a typed state |
| Tool | What it is for | Core abstraction |
| LlamaIndex | Data ingestion and retrieval for RAG: loaders, chunking, indexes, query engines | Documents → nodes → index → query engine |
| DSPy | Programming (not prompting) LLM pipelines; optimises prompts and few- shot examples against a metric | Signatures and modules, compiled by optimisers |
| Hugging Face | Models, datasets, tokenizers, training, PEFT, serving | `transformers`, `datasets`, `peft`, `trl`, `accelerate`, Hub |
| Provider SDKs | Direct calls to OpenAI, Anthropic, Google, etc. | Messages, tools, structured outputs |
| Instructor / Pydantic AI | Typed structured outputs and light agents | Pydantic models as the contract |
| Observability (LangSmith, Langfuse, Phoenix, MLflow tracing) | Traces, prompt versions, evaluations | Spans per LLM or tool call |

## 20.2 LangChain and LCEL

LangChain's useful core is small: a common interface (`invoke`, `batch`, `stream`, async variants) for models, prompts, retrievers and parsers, and the pipe operator to compose them.

```python
# Sketch, check current docs. Packages are split: langchain-core, langchain-
openai, ...
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

What it gives you: swapping providers by changing one object, batching and streaming without writing them, many integrations (vector stores, loaders), and tracing when paired with LangSmith. What it costs: deep abstraction layers that make errors hard to read, frequent breaking changes, and prompts hidden inside library defaults (always print the final prompt). Tools are declared with a decorator and bound to a model (`llm.bind_tools([...])`); structured output with `llm.with_structured_output(PydanticModel)`.

## 20.3 LangGraph

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

Use it when the flow has loops, branches, long-running state or approval steps (a support agent that can issue vouchers). For a fixed three-step pipeline it is overkill. Chapter 21 covers agent design itself.

## 20.4 LlamaIndex

Strongest at the data side of RAG: dozens of loaders (PDF, Notion, SQL), node parsers for chunking, index types (vector, keyword, summary, knowledge graph), retrievers with metadata filters, rerankers, and query engines that combine them (sub-question decomposition, router across indexes). A quick prototype is a few lines:

```python
# Sketch, check current docs
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
docs = SimpleDirectoryReader("policies/").load_data()
index = VectorStoreIndex.from_documents(docs)
answer = index.as_query_engine(similarity_top_k=5).query("What is the refund window for groceries?")
```

The defaults (chunk size, embedding model, prompt) are rarely right for production; you will end up configuring each piece, at which point the value is the integrations and the structure, not the magic.

## 20.5 DSPy

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

Good fit: a pipeline with a clear metric and a few hundred labelled examples, where you want to switch models without rewriting prompts by hand. Menu-item tagging against a fixed food taxonomy is a good example: the taxonomy gives you the label set and the metric. Weak fit: open-ended chat without a metric.

## 20.6 The Hugging Face stack

- `transformers`: models and tokenizers, `pipeline()` for quick inference, `Trainer` for fine-tuning (Chapter 12).
- `datasets`: memory-mapped datasets with fast `map` / `filter`, streaming for large corpora.
- `peft`: LoRA, QLoRA, prefix and prompt tuning (Chapter 16).
- `trl`: SFT, DPO, KTO, GRPO trainers (Chapter 13).
- `accelerate`: the same script on CPU, one GPU or many.
- `sentence-transformers`: embedding and cross-encoder models (Chapter 17).
- `text-generation-inference`, `text-embeddings-inference`: serving.
- The Hub: model cards, licences, versions. Pin a model by revision hash in production.

## 20.7 Framework or plain Python?

A decision rule worth saying in interviews:

- **Prototype or hackathon**: use the framework; speed matters more than control.
- **Production with a simple flow** (one retrieval, one call, structured output): provider SDK + Pydantic + your own small functions. Fewer dependencies, readable stack traces, no prompt surprises.
- **Production with complex control flow** (loops, approvals, long-running state): LangGraph or a workflow engine is worth it, but keep business logic in your own functions so you can leave the framework later.
- **Always**: own your prompts as versioned files, log every call (inputs, outputs, tokens, latency), and keep an evaluation set that runs in CI. These matter more than the framework choice.

Team fit counts too: if the platform team already supports one stack with tracing and deployment, the cost of a different choice is not just yours.

## Pitfalls

- Not knowing what prompt the framework actually sends. Log it.
- Pinning nothing: a minor version bump changes behaviour in production.
- Copying a tutorial's chunk size, embedding model and top-k into production without evaluation.
- Agent loops without a step limit or cost cap.
- Hard-wiring business rules inside framework callbacks, making migration painful.
- Choosing a framework because it is in the job ad rather than because it fits the problem.

## Interview questions

**Mid level**

1. What does LangChain's LCEL give you over calling the API directly?
2. What is the difference between LangChain and LangGraph?
3. When would you reach for LlamaIndex?
4. What does DSPy optimise?

**Senior level**

5. Your team's RAG service is built on LangChain and debugging is painful. Would you migrate? How would you decide and how would you do it safely?
6. Design a refund-handling agent with a human approval step. Which tool would you use and how does state persist across the pause?
7. How do you version, test and roll out a prompt change in a framework-based app?
8. How would you make framework choice a team decision rather than a personal preference?

Short answers: (5) Measure first: list the bugs and time lost; check what features you use. Migrate incrementally behind the same interface, with the evaluation set and traces comparing old and new on live traffic (shadow mode) before switching. (6) A LangGraph-style state machine with an interrupt before the refund node, a checkpointer keyed by conversation id, timeouts, and an audit log; the approver's decision resumes the graph. (7) Prompts as versioned files, CI evaluation with thresholds, canary or A/B rollout, traces tagged with the prompt version, quick rollback (Chapter 29).

## Practice

- Build the same small RAG over 50 policy documents three ways: plain Python + SDK, LangChain, LlamaIndex. Compare lines of code, latency, and how long it takes to find a planted bug.
- Re-implement the plain-Python graph with LangGraph and add a human-approval interrupt.
- Use DSPy to optimise a menu-item tagger on 200 labelled items and compare with your hand-written prompt.

## Read next

- LangChain documentation: LCEL and "How-to" guides; LangGraph documentation: concepts (state, checkpoints, interrupts).
- LlamaIndex documentation: "Building RAG from scratch" and the production RAG guides.
- Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023).
- Anthropic, "Building effective agents" (blog post, 2024): a strong argument for simple patterns over frameworks.
- Hugging Face course (free), chapters on Transformers, Datasets and fine-tuning.
- Links to the documentation of each tool are in the Key concepts section at the top of this chapter.

## Done when

You can describe what each major framework is for in one sentence.

You can write a small state-graph runner in plain Python and explain what LangGraph adds. You can write an LCEL chain and a LangGraph graph from memory, roughly.

You can argue framework vs plain Python for a given use case, with criteria. You have one project where you built the same thing with and without a framework.

---

[← 19 · Retrieval-augmented generation](19-retrieval-augmented-generation.md) · [Contents](../README.md#contents) · [21 · Agents and tool use →](21-agents-and-tool-use.md)
