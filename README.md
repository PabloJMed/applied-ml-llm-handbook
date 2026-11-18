# Applied ML & LLM Handbook

**Pablo Medina** · written with the guidance of Claude Opus 5.5 (Anthropic)

A self-study guide for data scientists and ML engineers preparing for senior applied ML and LLM roles: the theory, the code, the system design and the interview, in one place. Examples come from food delivery and marketplaces, but every chapter applies to any product company.

**Read it as a book:** [Applied-ML-and-LLM-Handbook.pdf](Applied-ML-and-LLM-Handbook.pdf) (249 pages, A4, with bookmarks and a linked contents page). Or browse the chapters below as Markdown.

## What is inside

- **35 chapters in 10 parts, plus 5 appendices:** foundations, deep learning, transformers, LLM training and serving, retrieval and RAG, agents, LLM evaluation, recommender and search systems, ML systems and production, and interview craft.
- **The same template in every chapter:** why it matters, key concepts with links to read further, intuition then maths then working code, diagrams, pitfalls, 10–20 interview questions with model answers, practice exercises and a "done when" checklist.
- **An infrastructure primer (Chapter 13a)** for readers new to production systems: servers, CPUs and GPUs, HTTP APIs, Docker, Kubernetes, queues, batch vs online inference, capacity planning and observability.
- **Appendices:** glossary, one-page cheat sheets, notation and formulas, a map of the papers worth knowing, and ten worked system designs.

New here? Start with the [Overview](chapters/00-overview.md): it explains how to use the handbook and gives a suggested 8-week study plan.

## Contents

**Part I · Foundations**

- [01 · The maths you actually use](chapters/01-the-maths-you-actually-use.md)
- [02 · Statistics and experimentation](chapters/02-statistics-and-experimentation.md)
- [03 · ML fundamentals](chapters/03-ml-fundamentals.md)
- [04 · Classic models](chapters/04-classic-models.md)
- [05 · Evaluation and metrics](chapters/05-evaluation-and-metrics.md)

**Part II · Deep learning**

- [06 · Neural network fundamentals](chapters/06-neural-network-fundamentals.md)
- [07 · PyTorch for practitioners](chapters/07-pytorch-for-practitioners.md)
- [08 · Embeddings and sequence models](chapters/08-embeddings-and-sequence-models.md)

**Part III · NLP and transformers**

- [09 · Text and tokenisation](chapters/09-text-and-tokenisation.md)
- [10 · The transformer in depth](chapters/10-the-transformer-in-depth.md)
- [11 · Model families and scaling](chapters/11-model-families-and-scaling.md)
- [12 · NLP tasks with modern tools](chapters/12-nlp-tasks-with-modern-tools.md)

**Part IV · Large language models**

- [13 · How LLMs are trained](chapters/13-how-llms-are-trained.md)
- [13a · Infrastructure primer](chapters/13a-infrastructure-primer.md)
- [14 · Inference, decoding and serving](chapters/14-inference-decoding-and-serving.md)
- [15 · Prompting and context engineering](chapters/15-prompting-and-context-engineering.md)
- [16 · Fine-tuning and LoRA](chapters/16-fine-tuning-and-lora.md)

**Part V · Retrieval and RAG**

- [17 · Embeddings](chapters/17-embeddings.md)
- [18 · Information retrieval](chapters/18-information-retrieval.md)
- [19 · Retrieval-augmented generation](chapters/19-retrieval-augmented-generation.md)

**Part VI · Agents**

- [20 · Frameworks and tooling](chapters/20-frameworks-and-tooling.md)
- [21 · Agents and tool use](chapters/21-agents-and-tool-use.md)

**Part VII · Evaluating LLM systems**

- [22 · LLM evaluation](chapters/22-llm-evaluation.md)
- [23 · RAG and agent evaluation](chapters/23-rag-and-agent-evaluation.md)
- [24 · Safety and responsible AI](chapters/24-safety-and-responsible-ai.md)

**Part VIII · Recommender and search systems**

- [25 · Recommender systems](chapters/25-recommender-systems.md)
- [26 · Search and learning to rank](chapters/26-search-and-learning-to-rank.md)
- [27 · Marketplace and logistics ML](chapters/27-marketplace-and-logistics-ml.md)

**Part IX · ML systems and production**

- [28 · ML system design](chapters/28-ml-system-design.md)
- [29 · MLOps and LLMOps](chapters/29-mlops-and-llmops.md)
- [30 · Data for ML](chapters/30-data-for-ml.md)

**Part X · Interview craft**

- [31 · The ML coding interview](chapters/31-the-ml-coding-interview.md)
- [32 · ML depth question bank](chapters/32-ml-depth-question-bank.md)
- [33 · Behavioural interviews](chapters/33-behavioural-interviews.md)
- [34 · Thinking like a senior](chapters/34-thinking-like-a-senior.md)

**Appendices**

- [A · Glossary](chapters/A-glossary.md)
- [B · Cheat sheets](chapters/B-cheat-sheets.md)
- [C · Notation and formulas](chapters/C-notation-and-formulas.md)
- [D · Paper map](chapters/D-paper-map.md)
- [E · Worked designs](chapters/E-worked-designs.md)

## Code

[`code/`](code/) has one runnable script per chapter that exercises the chapter's examples (from-scratch implementations checked against scikit-learn, PyTorch training loops, retrieval and ranking metrics, a FastAPI model server, and so on). Each script runs on a laptop CPU in seconds to a few minutes.

```bash
pip install -r code/requirements.txt
python code/ch04_check.py
```

## How this was checked

- Every script in `code/` ran end to end with Python 3.13, NumPy 2.5, scikit-learn 1.9 and PyTorch 2.15 (CPU).
- Every Python block in the chapters parses with Python's `ast` module; the longer examples are the ones exercised in `code/`.
- The links in the Key concepts sections were opened and checked in November 2026. Library APIs (LangChain, LlamaIndex, Hugging Face, vector databases) change every few months: if a snippet fails, check it against the current docs; the concepts stay the same.

## About

Written by Pablo Medina with the guidance of Claude Opus 5.5 (Anthropic), and reviewed and edited by the author. Corrections and suggestions are welcome: open an issue.

## Licence

Text, diagrams and the PDF: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Code in `code/`: [MIT](code/LICENSE). See [LICENSE](LICENSE).

## Edition

November 2026. The Markdown in `chapters/` is the single source: the PDF is built from it with [`tools/build_pdf.py`](tools/build_pdf.py), so they match.

```bash
pip install -r tools/requirements.txt
python tools/build_pdf.py
```

Fonts: Inter, Source Serif 4 and JetBrains Mono, all under the SIL Open Font License (licences in [`tools/fonts/`](tools/fonts/)).
