# The AI & LLM Scientist Handbook

**Pablo Medina & Claude Opus 5.5**

A self-study handbook for AI and LLM research scientists, and for the applied ML engineers who work alongside them: from the maths underneath to how large language models are pretrained, post-trained, served, evaluated and understood, with the research craft and interviews of the job. Theory, working code and worked numbers in one place.

**Read it as a book:** [AI-and-LLM-Scientist-Handbook.pdf](AI-and-LLM-Scientist-Handbook.pdf) (7 × 10 in book format, with bookmarks and a linked contents page; an A4 print version can be built, see below). Or browse the chapters below as Markdown.

## What is inside

- **48 chapters in 11 parts, plus 6 appendices.** Parts I–X are the core path for an LLM research scientist: foundations, deep learning and training dynamics, transformers and architectures, scaling laws, multimodal models, pretraining data and systems, post-training and reasoning, inference, evaluation science, interpretability, alignment and safety, retrieval and agents, research craft and interviews. Part XI is an optional applied track: classic models, recommender and search systems, ML system design, MLOps and data engineering.
- **The same template in every chapter:** why it matters, key concepts with links, intuition then maths then working code, pitfalls, interview questions with model answers, practice including a research exercise, further reading and a "done when" checklist.
- **Active recall:** the 250-question depth bank hides its answers until you click, with follow-up chains that go four levels deep and weak/acceptable/strong versions of the same answer. All 695 questions with answers are also exported as [flashcards](flashcards/) for Anki or any spaced-repetition app.
- **Numbers you can reproduce:** every chapter has a script in [`code/`](code/) whose output the text quotes, from a DPO run that recovers the analytic optimal policy to a tiny transformer whose induction circuit you can ablate.
- **Appendices:** a 40-question diagnostic that turns your gaps into a study plan, a glossary of 170+ terms, one-page cheat sheets (including LLM research), notation and formulas, a map of 119 papers from foundations to the end of 2025, and ten worked system designs.

New here? Start with the [Overview](chapters/00-overview.md): it explains how to use the handbook and gives a twelve-week research study plan, an eight-week applied plan and short refresh plans. Then take the [diagnostic](chapters/F-diagnostic.md) to see which chapters you need most.

## Contents

**Part I · Foundations**

- [1 · The maths you actually use](chapters/01-the-maths-you-actually-use.md)
- [2 · Statistics and experimentation](chapters/02-statistics-and-experimentation.md)
- [3 · ML fundamentals](chapters/03-ml-fundamentals.md)
- [4 · Evaluation and metrics](chapters/04-evaluation-and-metrics.md)

**Part II · Deep learning and training dynamics**

- [5 · Neural network fundamentals](chapters/05-neural-network-fundamentals.md)
- [6 · PyTorch for practitioners](chapters/06-pytorch-for-practitioners.md)
- [7 · Optimisation and training dynamics](chapters/07-optimisation-and-training-dynamics.md)
- [8 · Embeddings and sequence models](chapters/08-embeddings-and-sequence-models.md)

**Part III · Transformers and architectures**

- [9 · Text and tokenisation](chapters/09-text-and-tokenisation.md)
- [10 · The transformer in depth](chapters/10-the-transformer-in-depth.md)
- [11 · Architectures beyond the vanilla transformer](chapters/11-architectures-beyond-the-transformer.md)
- [12 · Model families and scaling](chapters/12-model-families-and-scaling.md)
- [13 · Scaling laws](chapters/13-scaling-laws.md)
- [14 · Multimodal models](chapters/14-multimodal-models.md)

**Part IV · Pretraining**

- [15 · How LLMs are trained](chapters/15-how-llms-are-trained.md)
- [16 · Pretraining data](chapters/16-pretraining-data.md)
- [17 · Pretraining at scale](chapters/17-pretraining-at-scale.md)

**Part V · Post-training**

- [18 · Post-training I: SFT, reward models and preference optimisation](chapters/18-post-training-sft-and-preferences.md)
- [19 · Post-training II: reinforcement learning and reasoning models](chapters/19-post-training-rl-and-reasoning.md)
- [20 · Fine-tuning and parameter-efficient methods](chapters/20-fine-tuning-and-lora.md)

**Part VI · Inference and prompting**

- [21 · Inference, decoding and serving](chapters/21-inference-decoding-and-serving.md)
- [22 · Prompting and context engineering](chapters/22-prompting-and-context-engineering.md)

**Part VII · Evaluation science**

- [23 · Evaluating models](chapters/23-evaluating-models.md)
- [24 · LLM evaluation](chapters/24-llm-evaluation.md)
- [25 · RAG and agent evaluation](chapters/25-rag-and-agent-evaluation.md)

**Part VIII · Interpretability, alignment and safety**

- [26 · Interpretability and analysis](chapters/26-interpretability.md)
- [27 · Alignment and safety research](chapters/27-alignment-and-safety-research.md)
- [28 · Safety and responsible AI](chapters/28-safety-and-responsible-ai.md)

**Part IX · Retrieval and agents**

- [29 · Embeddings and representation learning](chapters/29-embeddings.md)
- [30 · Information retrieval](chapters/30-information-retrieval.md)
- [31 · Retrieval-augmented generation](chapters/31-retrieval-augmented-generation.md)
- [32 · Frameworks and tooling](chapters/32-frameworks-and-tooling.md)
- [33 · Agents and tool use](chapters/33-agents-and-tool-use.md)

**Part X · Research craft and interviews**

- [34 · Research craft](chapters/34-research-craft.md)
- [35 · The ML coding interview](chapters/35-the-ml-coding-interview.md)
- [36 · Research interviews](chapters/36-research-interviews.md)
- [37 · ML depth question bank](chapters/37-ml-depth-question-bank.md)
- [38 · Behavioural interviews](chapters/38-behavioural-interviews.md)
- [39 · Thinking like a senior](chapters/39-thinking-like-a-senior.md)

**Part XI · Applied ML systems (optional track)**

- [40 · Classic models](chapters/40-classic-models.md)
- [41 · NLP tasks with modern tools](chapters/41-nlp-tasks-with-modern-tools.md)
- [42 · Infrastructure primer](chapters/42-infrastructure-primer.md)
- [43 · Recommender systems](chapters/43-recommender-systems.md)
- [44 · Search ranking and learning to rank](chapters/44-search-and-learning-to-rank.md)
- [45 · Marketplace and logistics ML](chapters/45-marketplace-and-logistics-ml.md)
- [46 · ML system design: the method](chapters/46-ml-system-design.md)
- [47 · MLOps and LLMOps](chapters/47-mlops-and-llmops.md)
- [48 · Data for ML](chapters/48-data-for-ml.md)

**Appendices**

- [A · Glossary](chapters/A-glossary.md)
- [B · Cheat sheets](chapters/B-cheat-sheets.md)
- [C · Notation and common formulas](chapters/C-notation-and-formulas.md)
- [D · Paper map](chapters/D-paper-map.md)
- [E · Worked system designs](chapters/E-worked-designs.md)
- [F · Diagnostic self-assessment](chapters/F-diagnostic.md)

## Code

[`code/`](code/) has a script for nearly every chapter that runs the chapter's examples and prints the numbers quoted in the text. Each runs on a laptop CPU in seconds to a few minutes.

```bash
pip install -r code/requirements.txt
python code/ch18_check.py
```

## How this was checked

- Every script in `code/` ran end to end with Python 3.12 and 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and PyTorch 2.15 (CPU) in November 2026; the exact versions are pinned in [`code/requirements.txt`](code/requirements.txt), and CI re-runs every script on each change (on macOS, two scripts need an OpenMP setting or `libomp`; see [`code/README.md`](code/README.md)).
- Every Python block in the chapters parses with Python's `ast` module; the longer examples are the ones exercised in `code/`.
- The links in the Key concepts sections point to papers' arXiv pages or official documentation. Library APIs change every few months: if a snippet fails, check it against the current docs; the concepts stay the same.

## About

By Pablo Medina & Claude Opus 5.5 (Anthropic). The second edition (November 2026) restructures the first edition, the *Applied ML & LLM Handbook*, around the LLM research-scientist role and adds 13 chapters; [CHANGELOG.md](CHANGELOG.md) lists what has changed since. Corrections and suggestions are welcome: open an issue.

## Checks

[![check](https://github.com/PabloJMed/applied-ml-llm-handbook/actions/workflows/check.yml/badge.svg)](https://github.com/PabloJMed/applied-ml-llm-handbook/actions/workflows/check.yml)

Every push and pull request runs [`tools/check_book.py`](tools/check_book.py) (Python blocks parse, relative links and anchors resolve, every chapter is in the contents and the PDF build), checks the flashcards are current and runs every script in `code/` with [`tools/run_code.py`](tools/run_code.py). Run the same checks locally before opening a pull request:

```bash
python tools/check_book.py
python tools/export_flashcards.py --check
python tools/run_code.py            # or a subset: python tools/run_code.py ch18 ch19
```

## Licence

Text, diagrams and the PDF: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Code in `code/`: [MIT](code/LICENSE). See [LICENSE](LICENSE). Fonts in `tools/fonts/`: SIL Open Font License.

## Building the PDF

The Markdown in `chapters/` is the single source: the PDF is built from it with [`tools/build_pdf.py`](tools/build_pdf.py), so they match.

```bash
uv run --python 3.12 --with-requirements tools/requirements.txt tools/build_pdf.py
```

Add `--page a4 --out AI-and-LLM-Scientist-Handbook-A4.pdf` for an A4 print version. Without uv: `pip install -r tools/requirements.txt` in a Python 3.10+ environment, then `python tools/build_pdf.py`.
