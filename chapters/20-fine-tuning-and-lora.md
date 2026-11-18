# 20 · Fine-tuning and parameter-efficient methods

Part V · Post-training. Assumes Chapter 10 (transformers) and Chapter 1's SVD and low-rank ideas. Chapter 15 covers how base models are trained and aligned; Chapter 24 how to evaluate the result.

## Why it matters

Fine-tuning adapts a pretrained model to your task, format or domain. Parameter-efficient methods, above all LoRA, made this possible on a single GPU and turned it into a routine tool: distilling an LLM's behaviour into a small model, enforcing a strict output format, teaching domain style or vocabulary, cutting cost and latency. Interviewers ask when fine-tuning is the right call (often it is not), how LoRA works mathematically, and how you would evaluate the result. Section 20.5 connects the older prompt-based methods (prefix-tuning and prompt tuning) to current practice.

## Key concepts

- **Full fine-tuning vs parameter-efficient fine-tuning (PEFT)**: updating every weight vs training a small number of new parameters on a frozen model. [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index).
- **LoRA**: learn two small low-rank matrices whose product is added to a frozen weight matrix; tiny to train and store, no extra latency once merged. [LoRA paper](https://arxiv.org/abs/2106.09685).
- **QLoRA**: LoRA on top of a 4-bit quantised base model, so large models fit on one GPU. [QLoRA paper](https://arxiv.org/abs/2305.14314).
- **Adapters, prefix-tuning and prompt tuning**: earlier PEFT methods that insert small modules or learn soft prompt vectors. [Adapters](https://arxiv.org/abs/1902.00751), [Prefix-tuning](https://arxiv.org/abs/2101.00190), [Prompt tuning](https://arxiv.org/abs/2104.08691).
- **Supervised fine-tuning (SFT) and preference tuning (DPO)**: training on input–output pairs, then on preferred vs rejected answers (Chapter 15). [TRL documentation](https://huggingface.co/docs/trl/index).
- **Catastrophic forgetting**: fine-tuning on a narrow task can erase general abilities. [Catastrophic interference](https://en.wikipedia.org/wiki/Catastrophic_interference).
- **When not to fine-tune**: if prompting or retrieval solves it, those are cheaper to build and change (20.1).

## 20.1 Prompt, retrieve or fine-tune?

| Need | Best first tool | Why |
|---|---|---|
| Use current, private or changing knowledge | RAG (Chapter 31) | Knowledge stays in an index you can update and cite; fine-tuning facts in is unreliable and stale |
| Follow instructions, format, tone | Prompting (Chapter 22) | Cheapest to iterate; often enough |
| Consistent strict format or style at scale | Fine-tuning (LoRA) | Shorter prompts, more reliable outputs |
| A narrow task at high volume, low latency or cost | Fine-tune a small model, often on LLM-generated labels | Distillation: most of the quality at a fraction of the cost |
| Domain language the base model handles poorly | Continued pre-training or fine-tuning on domain text | Changes the model's knowledge of the language itself |
| Better reasoning on a verifiable task | Reinforcement fine-tuning with verifiable rewards | Chapter 15 |

**The usual order:** prompt well, add retrieval if knowledge is missing, and fine-tune only when evaluation shows a consistent gap that prompting cannot close, or when cost and latency demand a smaller model. Fine-tuning brings costs of its own: data collection, training infrastructure, model hosting, re-training when the base model updates, and evaluation for regressions.

## 20.2 Full fine-tuning

Update every parameter on your data, usually with a small learning rate (around 1e-5 to 5e-5 for transformers), warm-up and decay, a few epochs, early stopping on validation loss.

**Memory is the obstacle.** Training with Adam in mixed precision needs roughly 16 bytes per parameter before activations: bf16 weights and gradients (4 bytes), an fp32 master copy (4) and Adam's two moment estimates (8).

```python
def finetune_memory_gb(n_params_billion, mode):
    n = n_params_billion * 1e9
    bytes_per_param = {"full_adam_mixed": 16,    # weights, grads, fp32 master copy, Adam m and v
                       "lora_bf16": 2,           # frozen bf16 base; adapters and their optimiser state are tiny
                       "qlora_4bit": 0.5}[mode]  # 4-bit frozen base
    return n * bytes_per_param / 1e9

finetune_memory_gb(7, "full_adam_mixed")   # 112 GB: several large GPUs, plus activations
finetune_memory_gb(7, "lora_bf16")         # 14 GB plus activations: one large GPU
finetune_memory_gb(7, "qlora_4bit")        # ~3.5 GB plus activations: one consumer GPU
```

Full fine-tuning is still right for smaller encoders (BERT-sized classifiers, Chapter 41.2), when you have lots of data and compute, or for large distribution shifts. For LLMs, parameter-efficient methods usually match it on narrow tasks at a fraction of the cost, and you keep one base model with many small adapters.

## 20.3 LoRA in depth

**The idea (Hu et al., 2021).** The weight change needed to adapt a model to a task has low intrinsic rank. So instead of learning a full update ΔW (d × k), learn it as the product of two thin matrices: ΔW = B·A, with B of size d × r and A of size r × k, r much smaller than d and k. The pretrained W stays frozen.

![Diagram](images/16-01.png)

*LoRA · frozen W plus a trainable low-rank path*

**The details that interviewers ask about:**

- **Initialisation:** A random, B zeros, so B·A = 0 at the start and the model begins exactly as the pretrained one.
- **Scaling α/r:** the update is multiplied by α/r. Setting α relative to r (often α = 2r, or α = r) keeps the update's size roughly stable when you change r, so you do not have to re-tune the learning rate. (rsLoRA proposes α/√r for high ranks.)
- **Rank r:** 8–64 is typical. Higher rank gives more capacity, more parameters and more risk of overfitting small data; returns diminish quickly.
- **Target modules:** the original paper adapted only the attention query and value projections; current practice usually adapts all linear layers (q, k, v, o and the MLP projections), which tends to work better for the same parameter budget spread across them.
- **Learning rate:** higher than full fine-tuning (around 1e-4 to 3e-4 is common), because few parameters move.
- **Merging:** after training, add (α/r)·B·A into W. The merged model has the same shape and speed as the original. Or keep adapters separate and swap them per request: one base model in memory serving many tasks (multi-LoRA serving in vLLM and similar engines).

**LoRA from scratch:**

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                               # freeze W (and bias)
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zeros: start as the base model
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            out = out + (self.dropout(x) @ self.A.T @ self.B.T) * self.scaling
        return out
    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.B @ self.A) * self.scaling
            self.merged = True
```

Checks from a test run: wrapping a 4096 × 4096 layer with r = 16 gives 131,072 trainable parameters out of 20.9 million (0.78%); on a toy task where the target weight differs from the base by a rank-4 change, a rank-4 LoRA drives the loss from 2.5 to about zero; and outputs before and after `merge()` are identical.

**Variants worth knowing:** DoRA (decomposes weights into magnitude and direction and applies LoRA to the direction), LoRA+ (different learning rates for A and B), and adapter merging methods for combining several LoRAs.

## 20.4 QLoRA

QLoRA (Dettmers et al., 2023) fine-tunes LoRA adapters on top of a base model **quantised to 4 bits**, so a 7B model's frozen weights take about 3.5 GB instead of 14 GB. Its ingredients:

- **NF4 (4-bit NormalFloat):** a 4-bit data type whose levels are spaced for normally distributed weights.
- **Double quantisation:** quantising the quantisation constants too, saving a bit more memory.
- **Paged optimisers:** moving optimiser state to CPU memory during spikes to avoid out-of-memory errors.

Weights are de-quantised to bf16 on the fly for each matrix multiplication; gradients flow through them into the bf16 LoRA adapters. Quality is close to 16-bit LoRA in the paper's experiments. The cost: slower training than bf16 LoRA. After training, merge the adapter into a 16-bit copy of the base model for serving, rather than into the 4-bit weights.

## 20.5 Adapters, prefix-tuning and prompt tuning

The PEFT family, roughly in historical order:

| Method | What is trained | Notes |
|---|---|---|
| Adapters (Houlsby et al., 2019) | Small bottleneck MLPs inserted after attention and FFN in each layer | Effective; adds inference latency because extra layers run in sequence |
| Prefix-tuning (Li and Liang, 2021) | Trainable key and value vectors prepended at every layer | Strong for generation tasks; often reparameterised with an MLP for stable training |
| Prompt tuning (Lester et al., 2021) | A few trainable embedding vectors (a soft prompt) prepended to the input only | Fewest parameters; works well for very large models, weaker for small ones |
| P-tuning v2 (Liu et al., 2021) | Deep prompt tuning at every layer, like prefix-tuning, for understanding tasks | Narrows the gap to fine-tuning on smaller models |
| LoRA (2021) and successors | Low-rank updates to existing weights | No added latency after merging; the current default |
| IA³ | Learned vectors rescaling activations | Extremely few parameters |

**Prompt tuning in code:**

```python
class SoftPrompt(nn.Module):
    """n_tokens trainable vectors prepended to a frozen model's input embeddings."""
    def __init__(self, n_tokens, d_model, init_from=None):
        super().__init__()
        init = (init_from[:n_tokens].clone() if init_from is not None      # e.g. embeddings of real words
                else torch.randn(n_tokens, d_model) * 0.02)
        self.prompt = nn.Parameter(init)

    def forward(self, token_embeds):  # (B, T, d)
        B = token_embeds.shape[0]
        return torch.cat([self.prompt.unsqueeze(0).expand(B, -1, -1), token_embeds], dim=1)
```

Initialising the soft prompt from real token embeddings (for example the class names) trains faster and more stably than random initialisation.

**Prompt-based methods in context.** Prefix-tuning and prompt tuning are the conceptual ancestors of today's PEFT: freeze the model, learn a small set of parameters that steer it. In an interview, be ready to explain why LoRA largely replaced these methods in practice:

it is more stable to train, works well across model sizes, and adds no inference latency once merged. If you have used these methods in research or a side project, this comparison turns that experience into evidence of depth.

## 20.6 Instruction data and chat templates

**Supervised fine-tuning (SFT)** trains on (instruction, response) pairs, or multi-turn conversations. Quality beats quantity: the LIMA paper (Zhou et al., 2023) showed strong instruction-following from about 1,000 carefully chosen examples. For a narrow task, a few hundred to a few thousand high-quality examples is a common range.

**What good task data looks like:** realistic inputs (from production where possible), covering the full distribution and the hard cases; correct, consistent outputs in exactly the target format; deduplicated; no overlap with your evaluation set.

**Chat templates.** Instruction-tuned models expect a specific format with role markers and special tokens. Use the tokenizer's own template (`tokenizer.apply_chat_template`) for training and inference; a mismatch silently degrades quality.

**Loss masking.** Compute the loss only on the assistant's response tokens, not on the prompt, so the model learns to answer rather than to reproduce prompts. Most SFT libraries (for example TRL's `SFTTrainer`) support completion-only loss.

## 20.7 Synthetic data

LLM-generated training data is now standard, especially for distillation:

- **Distillation:** a strong model labels or answers real inputs; a small model is fine-tuned on its outputs (Chapter 41.1's label-and-distil pattern). Check the provider's terms on using outputs to train models.
- **Generating inputs:** varied by persona, intent, difficulty and language, seeded with real examples. Generated inputs are cleaner and less diverse than real ones; mix in real data and deduplicate near-duplicates.
- **Filtering:** keep only examples that pass checks (format validation, an LLM judge, agreement between several generations, execution for code).
- **Risks:** copying the teacher's errors and biases, low diversity, and "model collapse" when models train repeatedly on model outputs without fresh real data.

Always keep a human-verified evaluation set that the synthetic pipeline never touched.

## 20.8 Catastrophic forgetting

Fine-tuning on a narrow task can degrade general abilities, including safety behaviour and instruction-following outside the task. Mitigations:

- PEFT methods, which change fewer parameters and tend to forget less (Biderman et al., 2024, found LoRA "learns less and forgets less" than full fine-tuning).
- Lower learning rates and fewer epochs; early stopping on a general-ability check, not only task loss.
- Mixing some general instruction data into the training set (replay).
- Keeping the adapter separate and only using it for the task's traffic.
- Evaluating a small general and safety suite before and after every fine-tune.

## 20.9 Evaluating a fine-tune

1. **Baselines:** the base model with your best prompt, the base model with few-shot examples, and possibly a larger model. The fine-tune must beat these on the task by enough to justify itself.
2. **Task metrics** on a held-out set never used for training or prompt tuning (Chapter 24), per slice.
3. **Format reliability:** share of outputs that parse and validate.
4. **Regression suite:** general abilities, safety refusals, and other tasks the model must still do.
5. **Overfitting checks:** training vs validation loss curves; outputs on novel inputs rather than near-duplicates of training data.
6. **Cost and latency** compared with the prompted alternative.

## 20.10 Hugging Face PEFT in practice

A QLoRA setup with Transformers, PEFT and bitsandbytes (API as of 2025–26; library details change, so check current docs):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "<an open instruction-tuned model>"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])   # names depend on the architecture
model = get_peft_model(model, config)
model.print_trainable_parameters()       # typically well under 1% of the total

# Train with TRL's SFTTrainer (chat-formatted dataset, completion-only loss),
# then: model.save_pretrained("adapter/")  -> a few tens of MB
# For serving: load the base in bf16, attach the adapter, merge_and_unload(), or serve multi-LoRA.
```

Managed alternatives: most LLM providers offer fine-tuning APIs for some of their models (upload JSONL conversations, get a fine-tuned model id), which removes infrastructure work at the cost of control.

## 20.11 Beyond task fine-tuning: continued pretraining, merging and RL fine-tuning

**Continued pretraining (domain adaptation).** To teach a model a new domain or language rather than a task, continue next-token training on raw domain text (billions of tokens), then redo instruction tuning. Mix in 10–30% general data to limit forgetting, re-warm the learning rate gently (a sudden high rate damages the model; a warm-up and decay similar to the original schedule's tail works better), and evaluate general abilities as well as the domain. LoRA learns less and forgets less than full fine-tuning here; for large knowledge injections full fine-tuning is usually needed.

**Model merging.** Fine-tuned models that share a base can be combined in weight space without further training:

- **Weight averaging ("model soups")**: average several fine-tunes of the same base trained with different hyperparameters; often better and more robust than the best single one.
- **Task arithmetic**: a task vector is the difference between fine-tuned and base weights; adding several task vectors (scaled) to the base combines skills, subtracting one removes a behaviour.
- **TIES and DARE**: reduce interference between task vectors by trimming small changes, resolving sign conflicts (TIES) or randomly dropping and rescaling changes (DARE) before adding them.

Merging is cheap and widely used to combine specialised fine-tunes or checkpoints from different training runs; its results are empirical and need the same evaluation as any other model.

**Reinforcement fine-tuning.** For tasks with checkable outcomes (structured extraction with a validator, code with tests, maths, classification with clear rubrics), fine-tuning with RL against a grader (Chapter 19) can beat SFT, especially with few labelled examples, because the model learns from its own attempts. Several providers offer it as a managed service with user-defined graders; the grader then becomes the specification and needs the same hardening as any verifier (Chapter 27).

## Pitfalls

- **Fine-tuning to add knowledge** that changes or must be cited; use RAG.
- **Skipping the prompted baseline,** so nobody knows whether fine-tuning helped.
- **Wrong chat template** or training on prompt tokens.
- **Evaluation data leaking into training,** especially with synthetic pipelines seeded from the evaluation set.
- **Too high a learning rate or too many epochs** on small data: memorisation and forgetting.
- **Changing r without adjusting α** (or the learning rate).
- **Merging into 4-bit weights** instead of a 16-bit copy of the base.
- **No general-ability and safety regression checks.**
- **Ignoring the maintenance cost:** every base-model upgrade means re-training and re-evaluating.

## Interview questions

Questions 1–4 basic, 5–9 intermediate, 10–13 senior.

**1. When would you fine-tune instead of prompting or using RAG?** For consistent format or style at scale, narrow high-volume tasks where a small fine-tuned model replaces an expensive one, or domain language the base model handles poorly. Not for adding changing facts (RAG) or when prompting already meets the bar.

**2. Explain LoRA.** Freeze W; learn ΔW = B·A with rank r much smaller than the dimensions; output Wx + (α/r)BAx. B starts at zero so training starts from the base model. Trains well under 1% of parameters; merge afterwards for zero latency cost.

**3. What do rank and alpha do?** Rank sets the capacity of the update; alpha scales it (by α/r), keeping the update's magnitude comparable across ranks so learning rates transfer.

**4. What is QLoRA?** LoRA on a frozen base quantised to 4-bit NF4 with double quantisation and paged optimisers, so large models fine-tune on one GPU with quality close to 16-bit LoRA.

**5. Prompt tuning vs prefix-tuning vs LoRA?** Prompt tuning learns soft input embeddings only; prefix-tuning learns key/value prefixes at every layer; LoRA learns low-rank updates to weights. LoRA is the most robust across model sizes and adds no inference latency once merged; prompt tuning is very parameter-cheap but weaker on small models.

**6. Which modules would you target with LoRA, and why?** All linear projections in attention and MLP blocks usually work best for a given budget; attention-only (q, v) is the original minimal setup. Choose by evaluation.

**7. How much memory does fine-tuning a 7B model need?** Full fine-tuning with Adam in mixed precision: about 16 bytes per parameter, roughly 112 GB plus activations. LoRA on a bf16 base: about 14 GB plus activations. QLoRA: about 3.5 GB for weights plus activations and adapters.

**8. What is catastrophic forgetting and how do you limit it?** Loss of general abilities after narrow fine-tuning. Limit with PEFT, low learning rates, fewer epochs, replay of general data, and a regression suite.

**9. How do you build training data for a fine-tune?** Real inputs where possible, outputs from experts or a strong model, filtered for quality and format, deduplicated, balanced across cases, with a separate human-verified evaluation set.

**10. You need a classifier for 50 million merchant descriptions a month. Plan it.** Label a stratified sample with a strong LLM (batch API), audit a few hundred by hand, fine-tune a small encoder or a small LLM with LoRA on the LLM labels, compare with TF-IDF and prompted baselines, route low-confidence cases to the LLM, monitor drift and re-label periodically.

**11. Your fine-tuned model scores higher on the task but users complain it got worse. What happened?** The evaluation set may not represent real traffic; general abilities or tone regressed; it overfits training phrasing; the chat template or decoding settings differ in production. Compare on production samples and run the regression suite.

**12. How would you serve 20 different LoRA adapters efficiently?** One base model in GPU memory with multi-LoRA serving (adapters loaded per request and batched together), adapters cached by popularity, merged models only for the highest-traffic adapters.

**13. Why might LoRA match full fine-tuning on some tasks and not others?** Narrow tasks need low-rank changes, which LoRA captures; large distribution shifts (new languages, continued pre-training on lots of domain text, complex skills) need higher-rank updates, where full fine-tuning or high-rank LoRA does better.

## Practice

1. **LoRA from scratch:** wrap the linear layers of the mini-GPT from Chapter 10 with `LoRALinear`, fine-tune on a new text style, and confirm merge equivalence.
2. **Prompt tuning from scratch:** freeze the same mini-GPT and train a 10-token soft prompt for the new style; compare with LoRA at a similar parameter count.
3. **PEFT on a real model:** fine-tune a small open model (1–4B parameters) with QLoRA on a classification-as-generation task (for example your merchant categories on public data) on a free or cheap GPU; compare with the prompted base model on a held-out set.
4. **Rank sweep:** r = 4, 16, 64 with α = 2r; plot task quality and training time.
5. **Forgetting check:** measure a small general-ability set before and after fine-tuning; then add 10% general instruction data and re-measure.
6. **Distillation:** label 2,000 examples with a large model, fine-tune a small one, and compute cost per 1,000 predictions for both.

## Read next

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023).
- Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning* (2021); Li and Liang, *Prefix-Tuning* (2021); Houlsby et al., *Parameter-Efficient Transfer Learning for* *NLP* (2019).
- Biderman et al., *LoRA Learns Less and Forgets Less* (2024).
- Zhou et al., *LIMA: Less Is More for Alignment* (2023).
- Hugging Face PEFT and TRL documentation.
- Sebastian Raschka's articles on LoRA and fine-tuning experiments, and *Build a Large* *Language Model (From Scratch)*, chapters on fine-tuning.
- *AI Engineering* (Chip Huyen): the chapters on finetuning and dataset engineering.

**Links:** [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [PEFT docs](https://huggingface.co/docs/peft/index) · [TRL docs](https://huggingface.co/docs/trl/index). Other titles above can be found by searching their title.

## Done when

You can decide between prompting, RAG and fine-tuning for a given need and defend it. You can write LoRA from scratch, including initialisation, scaling and merging. You can explain rank, alpha, target modules and learning rates for LoRA. You can explain QLoRA's components and do the memory maths. You can compare adapters, prefix-tuning, prompt tuning and LoRA, and explain why LoRA won in practice. You can design instruction data, use chat templates and loss masking. You can evaluate a fine-tune against baselines with a regression suite.

---

[← 19 · Post-training II: reinforcement learning and reasoning models](19-post-training-rl-and-reasoning.md) · [Contents](../README.md#contents) · [21 · Inference, decoding and serving →](21-inference-decoding-and-serving.md)
