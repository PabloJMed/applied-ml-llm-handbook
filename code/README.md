# Code

One script per chapter, plus one for the worked designs in Appendix E. Each runs the chapter's code examples end to end and prints the numbers quoted in the text. Run any of them with `python code/<script>.py` after `pip install -r code/requirements.txt`. All run on a laptop CPU; most take seconds; the longest (chapters 10, 14, 26 and 35, which train small models) take several minutes. `python tools/run_code.py` runs them all.

| Script | Chapter | Covers |
|---|---|---|
| [`ch01_check.py`](ch01_check.py) | [1 · The maths you actually use](../chapters/01-the-maths-you-actually-use.md) | The maths you actually use |
| [`ch02_check.py`](ch02_check.py) | [2 · Statistics and experimentation](../chapters/02-statistics-and-experimentation.md) | Statistics and experimentation |
| [`ch03_check.py`](ch03_check.py) | [3 · ML fundamentals](../chapters/03-ml-fundamentals.md) | ML fundamentals |
| [`ch04_check.py`](ch04_check.py) | [4 · Evaluation and metrics](../chapters/04-evaluation-and-metrics.md) | Evaluation and metrics |
| [`ch05_check.py`](ch05_check.py) | [5 · Neural network fundamentals](../chapters/05-neural-network-fundamentals.md) | Neural network fundamentals |
| [`ch06_check.py`](ch06_check.py) | [6 · PyTorch for practitioners](../chapters/06-pytorch-for-practitioners.md) | PyTorch for practitioners |
| [`ch07_check.py`](ch07_check.py) | [7 · Optimisation and training dynamics](../chapters/07-optimisation-and-training-dynamics.md) | Optimisation and training dynamics |
| [`ch08_check.py`](ch08_check.py) | [8 · Embeddings and sequence models](../chapters/08-embeddings-and-sequence-models.md) | Embeddings and sequence models |
| [`ch09_check.py`](ch09_check.py) | [9 · Text and tokenisation](../chapters/09-text-and-tokenisation.md) | Text and tokenisation |
| [`ch10_check.py`](ch10_check.py) | [10 · The transformer in depth](../chapters/10-the-transformer-in-depth.md) | The transformer in depth |
| [`ch11_check.py`](ch11_check.py) | [11 · Architectures beyond the vanilla transformer](../chapters/11-architectures-beyond-the-transformer.md) | Architectures beyond the vanilla transformer |
| [`ch12_check.py`](ch12_check.py) | [12 · Model families and scaling](../chapters/12-model-families-and-scaling.md) | Model families and scaling |
| [`ch13_check.py`](ch13_check.py) | [13 · Scaling laws](../chapters/13-scaling-laws.md) | Scaling laws |
| [`ch14_check.py`](ch14_check.py) | [14 · Multimodal models](../chapters/14-multimodal-models.md) | Multimodal models |
| [`ch15_check.py`](ch15_check.py) | [15 · How LLMs are trained](../chapters/15-how-llms-are-trained.md) | How LLMs are trained |
| [`ch16_check.py`](ch16_check.py) | [16 · Pretraining data](../chapters/16-pretraining-data.md) | Pretraining data |
| [`ch17_check.py`](ch17_check.py) | [17 · Pretraining at scale](../chapters/17-pretraining-at-scale.md) | Pretraining at scale |
| [`ch18_check.py`](ch18_check.py) | [18 · Post-training I: SFT, reward models and preference optimisation](../chapters/18-post-training-sft-and-preferences.md) | Post-training I: SFT, reward models and preference optimisation |
| [`ch19_check.py`](ch19_check.py) | [19 · Post-training II: reinforcement learning and reasoning models](../chapters/19-post-training-rl-and-reasoning.md) | Post-training II: reinforcement learning and reasoning models |
| [`ch20_check.py`](ch20_check.py) | [20 · Fine-tuning and parameter-efficient methods](../chapters/20-fine-tuning-and-lora.md) | Fine-tuning and parameter-efficient methods |
| [`ch21_check.py`](ch21_check.py) | [21 · Inference, decoding and serving](../chapters/21-inference-decoding-and-serving.md) | Inference, decoding and serving |
| [`ch22_check.py`](ch22_check.py) | [22 · Prompting and context engineering](../chapters/22-prompting-and-context-engineering.md) | Prompting and context engineering |
| [`ch23_check.py`](ch23_check.py) | [23 · Evaluating models](../chapters/23-evaluating-models.md) | Evaluating models |
| [`ch24_check.py`](ch24_check.py) | [24 · LLM evaluation](../chapters/24-llm-evaluation.md) | LLM evaluation |
| [`ch25_check.py`](ch25_check.py) | [25 · RAG and agent evaluation](../chapters/25-rag-and-agent-evaluation.md) | RAG and agent evaluation |
| [`ch26_check.py`](ch26_check.py) | [26 · Interpretability and analysis](../chapters/26-interpretability.md) | Interpretability and analysis |
| [`ch27_check.py`](ch27_check.py) | [27 · Alignment and safety research](../chapters/27-alignment-and-safety-research.md) | Alignment and safety research |
| [`ch28_check.py`](ch28_check.py) | [28 · Safety and responsible AI](../chapters/28-safety-and-responsible-ai.md) | Safety and responsible AI |
| [`ch29_check.py`](ch29_check.py) | [29 · Embeddings and representation learning](../chapters/29-embeddings.md) | Embeddings and representation learning |
| [`ch30_check.py`](ch30_check.py) | [30 · Information retrieval](../chapters/30-information-retrieval.md) | Information retrieval |
| [`ch31_check.py`](ch31_check.py) | [31 · Retrieval-augmented generation](../chapters/31-retrieval-augmented-generation.md) | Retrieval-augmented generation |
| [`ch32_check.py`](ch32_check.py) | [32 · Frameworks and tooling](../chapters/32-frameworks-and-tooling.md) | Frameworks and tooling |
| [`ch33_check.py`](ch33_check.py) | [33 · Agents and tool use](../chapters/33-agents-and-tool-use.md) | Agents and tool use |
| [`ch34_check.py`](ch34_check.py) | [34 · Research craft](../chapters/34-research-craft.md) | Research craft |
| [`ch35_check.py`](ch35_check.py) | [35 · The ML coding interview](../chapters/35-the-ml-coding-interview.md) | The ML coding interview |
| [`ch36_check.py`](ch36_check.py) | [36 · Research interviews](../chapters/36-research-interviews.md) | Research interviews: budgets, the DPO loss at initialisation |
| [`ch37_check.py`](ch37_check.py) | [37 · ML depth question bank](../chapters/37-ml-depth-question-bank.md) | Numbers quoted in the question bank answers |
| [`ch38_check.py`](ch38_check.py) | [38 · Behavioural interviews](../chapters/38-behavioural-interviews.md) | The seed-noise numbers in the research story |
| [`ch39_check.py`](ch39_check.py) | [39 · Thinking like a senior](../chapters/39-thinking-like-a-senior.md) | Scoping and triage arithmetic |
| [`ch40_check.py`](ch40_check.py) | [40 · Classic models](../chapters/40-classic-models.md) | Classic models |
| [`ch40_svm_check.py`](ch40_svm_check.py) | [40 · Classic models](../chapters/40-classic-models.md) | SVM from scratch vs scikit-learn |
| [`ch41_check.py`](ch41_check.py) | [41 · NLP tasks with modern tools](../chapters/41-nlp-tasks-with-modern-tools.md) | NLP tasks with modern tools |
| [`ch42_check.py`](ch42_check.py) | [42 · Infrastructure primer](../chapters/42-infrastructure-primer.md) | Infrastructure primer |
| [`ch43_check.py`](ch43_check.py) | [43 · Recommender systems](../chapters/43-recommender-systems.md) | Recommender systems |
| [`ch44_check.py`](ch44_check.py) | [44 · Search ranking and learning to rank](../chapters/44-search-and-learning-to-rank.md) | Search ranking and learning to rank |
| [`ch45_check.py`](ch45_check.py) | [45 · Marketplace and logistics ML](../chapters/45-marketplace-and-logistics-ml.md) | Marketplace and logistics ML |
| [`ch46_check.py`](ch46_check.py) | [46 · ML system design](../chapters/46-ml-system-design.md) | Training-run and LLM-serving design arithmetic |
| [`ch47_check.py`](ch47_check.py) | [47 · MLOps and LLMOps](../chapters/47-mlops-and-llmops.md) | MLOps and LLMOps |
| [`ch48_check.py`](ch48_check.py) | [48 · Data for ML](../chapters/48-data-for-ml.md) | Data for ML |
| [`appE_check.py`](appE_check.py) | [E · Worked system designs](../chapters/E-worked-designs.md) | Numbers in the worked designs |

**macOS notes.** LightGBM (used by `ch43_check.py` and `ch45_check.py`) needs the OpenMP runtime: `brew install libomp`. `ch30_check.py` imports both PyTorch and faiss, which bundle different OpenMP runtimes; run it as `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python code/ch30_check.py`.
