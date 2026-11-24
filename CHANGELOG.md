# Changelog

## Review edition: deeper derivations, full answers, frontier topics (November 2026)

A critical review of the whole book, read as a junior-to-mid scientist preparing for senior interviews, found four systemic gaps: interview questions without model answers, results stated rather than derived exactly where interviewers ask "derive it", practice that could not be self-checked, and thin coverage of 2025–26 topics. This edition closes them chapter by chapter.

**Template promises kept**
- Every interview question in every chapter now has a model answer: about 50 had none (Ch 6, 8, 9, 12, 15, 21, 28, 32, 39, 42, 47, 48), and Ch 35 and 38 gained an interview-question section. Answers are rewritten to the "strong" shape: the core point, one number computed on the spot, a trade-off or when not to, and the follow-up you will get. Chapter interview questions go from 405 to 679, the question bank from 250 to 260, the diagnostic from 40 to 48; the flashcards from 695 to 987.
- Every chapter now has a research exercise (14 did before), every practice item a hint and an expected result, and every "Done when" list concrete, timed tasks with numeric answers.
- Display equations go from 42 to 272, each with its symbols explained and a worked example.

**Derivations added where interviewers drill** (a selection): MSE as Gaussian MLE and ridge as MAP; the score-function trick, importance sampling and KL estimators; the delta method, CUPED, sequential boundaries and AIPW; bias–variance with the cross terms; the softmax–cross-entropy gradient; residual connections as a gradient highway (new §5.8); the gradient noise scale; Muon and Newton–Schulz; FlashAttention's IO argument; 6ND from one linear layer and what it omits; the attention/FFN crossover at T ≈ 6d; Mamba/SSMs and the delta rule; the MoE auxiliary loss; the Muennighoff effective-data formula; flow matching and the DDPM sampler; MinHash ≈ Jaccard and the LSH threshold; ZeRO and tensor-parallel communication costs; DPO in full with its gradient, and the SimPO, IPO, KTO and ORPO losses; the full GRPO objective; GAE λ; why LoRA initialises B to zero; speculative decoding's losslessness and wall-clock speed-up; GPTQ, AWQ and SmoothQuant; paired and clustered standard errors and Bradley–Terry; Rogan–Gladen with uncertainty; the residual stream, QK/OV circuits and the SAE loss; PGR; InfoNCE's gradient; HNSW and BM25's IDF; XGBoost's gain, bagging variance and the SVM dual; LambdaRank's λ; Erlang C; PSI as a symmetric KL.

**New and expanded topics** (a selection): double descent, grokking and distribution shift (Ch 3); an "In LLM work" bridge for Part I; teacher forcing, attention sinks and the modern-vs-2017 transformer (Ch 10); a 2025 open-model landscape (Ch 12, now a map with pointers instead of duplicates); why power laws exist and the Besiroglu refit (Ch 13); grounding, GUI agents and native multimodality (Ch 14); NVL72-era hardware, fp8 training and an MFU debugging walkthrough (Ch 17); preference-data collection, rejection sampling and online vs offline (Ch 18); hybrid thinking, R1's stages and RL cost models (Ch 19); LoRA Without Regret and safety erosion from fine-tuning (Ch 20); serving at scale (Ch 21); context engineering for agents and a prompt-injection threat model with the lethal trifecta and CaMeL (Ch 22, 28, 33); choosing evals for small-scale ablations and within-question sampling variance (Ch 23); multi-turn and two-sided safety evaluation (Ch 24); hardening agent graders against exploits (Ch 25); attribution methods, DAS and crosscoders (Ch 26); Constitutional AI, model specs and deliberative alignment (Ch 27); fairness definitions and their impossibility, memorisation, membership inference, DP-SGD, unlearning and watermarking (Ch 28); research-side RAG (Ch 31); agentic RL and coding agents (Ch 33); a debugging round, MHA with GQA and a KV cache, and AI-assisted rounds (Ch 35); the research-vision round (Ch 36); research STAR stories, senior signals from mid-sized scope and frontier-lab mission questions (Ch 38); research scoping, run triage and a mid/senior/staff rubric (Ch 39); embedding tables at scale and calibration after negative downsampling (Ch 43); an LLM-serving design and a fully worked training-run design (Ch 46); sequential canary tests and research-scale operations (Ch 47); preference annotation (Ch 48).

**Appendices.** The glossary (217 terms) points every term at the chapter that covers it in depth; the cheat sheets add post-training objectives and an inference-numbers sheet; notation adds the full GRPO, PPO and DPO-gradient forms and the modern decoder block; the paper map (126 papers) adds a "Know cold" note to every core paper and the missing classics; the worked designs add five lab designs (LLM serving, an RLVR pipeline, a pretraining data pipeline, an evaluation platform, a coding agent); the diagnostic adds eight questions, partial-credit rules and a parallel retake form.

**Corrections** (a selection): LoRA's parameter count (20.9M → 16.9M); speculative decoding's acceptance rule in the question bank (it was only right for greedy decoding); the GRPO code in Ch 35 (per-group baseline, correct DAPO/Dr. GRPO naming); a pointwise IPS loss that was biased (Ch 43); a Dockerfile that would not build (Ch 42); the LSH threshold (0.75 → 0.71); a benchmark standard error (2.2 → 2.1 points); an RNN gradient ratio ("a million" → about 200,000); a leakage test that did not test leakage (Ch 48); class weights that broke the calibration a later step relied on, and a prompt-cost estimate that could not hold its taxonomy (Ch 41); best-of-n KL stated as an equality (it is an upper bound); the confidence-interval wording in Ch 39; several wrong chapter cross-references.

**Tools.** `tools/check_book.py` now fails if a chapter question lacks a model answer or Practice lacks a research exercise; `tools/check_equations.py` typesets every display equation as the PDF build does and runs in CI; `tools/run_code.py` also runs `code/appE_check.py`; figure labels for sample sizes and break-evens are computed in closed form; hand-drawn diagrams are renamed after the chapter that uses them (`NN-diagram-K.png`). New check scripts for chapters 36–39, 46 and Appendix E.

## Front page and 7 pt body text (November 2026)

- New front page: dark background with a causal attention matrix as the motif, the title, subtitle, authors and edition only; the list of parts is gone from the cover (the contents page is unchanged).
- Body text 7 pt (was 7.4). The book is 386 pages.

## Smaller body text (November 2026)

- Body 7.4 pt (was 8), after a second printed proof; headings, tables and equations scale with it. Code blocks are set relatively a little larger (0.76 of the body size, was 0.72) so they stay readable.

## Print format and deeper explanations (November 2026)

**Format**
- Smaller type for print: body 8 pt (was 8.6), headings scaled down with it. The 7 × 10 in page is enlarged about 1.18× when printed on A4, so this reads like 9.4 pt on paper.
- More space after headings and between paragraphs (line height 1.4, gap 0.95).
- Tables sit on a light-blue panel with a blue header rule and white row separators, replacing the heavy black rules.
- Pitfalls, Interview questions, Practice and Read next each get their own coloured block, like Key concepts.

**Content**
- Chapter 1 rewritten towards a textbook: a fuller "Why it matters", an opening paragraph for every section, and section 1.6 (entropy, cross-entropy, KL divergence) rebuilt from first principles with worked numbers. `tools/writing-guide.md` sets this as the target depth for the other chapters.
- Part I (chapters 2–4) expanded the same way: longer chapter openings that explain how the sections fit together, an opening paragraph for every section, and deeper explanations where the text was a list of one-liners (the central limit theorem, a p-value worked by hand, power formulas as display equations, the arithmetic of multiple testing, how interleaving works, confounding, regularisation as a prior, proper scoring rules, MAE/RMSE/WAPE as display formulas).
- Part II (chapters 5–8) expanded the same way, with the MLP forward pass, AdamW, the gradient noise scale and the RNN and LSTM updates as display equations, and openings that explain why each technique exists (initialisation, normalisation, schedules, number formats, recurrence, attention).
- Part III (chapters 9–14): chapter openings that lay out how the sections build on each other, an opening paragraph per section naming the problem it solves, and RMSNorm, SwiGLU and the Chinchilla loss as display equations.
- Parts IV and V (chapters 15–20): the same treatment, with 6ND, the GRPO advantage and the LoRA update as display equations, and chapter openings that name the idea tying each chapter together (each training stage changes something different; every parallelism scheme trades memory for communication; post-training maximises reward under a KL leash; a narrow adaptation is a low-rank change).
- Parts VI–VIII (chapters 21–28): chapter openings that lay out the sections and their unifying idea (serving is bound by memory reads; a prompt is a specification; evaluation is measurement; interpretability needs interventions; alignment failures are optimisers exploiting objectives; safety is defence in depth), section openings where sections started cold, and the benchmark standard error as a display equation.
- Parts IX–X (chapters 29–39, except the question bank in 37): chapter openings that lay out the sections and their unifying idea, section openings where sections started cold, BM25 as a display equation, and prerequisite lines for chapters 32 and 39.
- Part XI (chapters 40–48): chapter openings that lay out the sections and their unifying idea, the normal equation, the boosting update and the SVM objective as display equations, and prerequisite lines for chapters 46 and 48.
- `tools/check_book.py` now also checks that every `$$` block has a blank line before and after it, and `tools/build_pdf.py` stops if an equation was not typeset.

## Typeset display equations (November 2026)

- Key equations are now set as numbered display formulas, centred with real fractions, sums and roots, instead of plain-text lines: CUPED (2), bias-variance (3), log loss (4), attention (10), reward-model, RLHF, PPO and DPO objectives (18), InfoNCE (29) and the matrix-factorisation and off-policy losses (43).
- Source format: a `$$ ... $$` block of LaTeX on its own lines. GitHub renders it directly; `tools/build_pdf.py` typesets it with matplotlib mathtext (STIX fonts) and numbers it per chapter (Eq. 10.1). `tools/check_book.py` checks the delimiters.
- `tools/requirements.txt` now lists matplotlib.

## PDF layout: denser book format (November 2026)

- Body text 8.6 pt (was 9 pt), line height 1.36 (was 1.42), 15% less space between blocks, slightly narrower margins and figures at 72% of the text width (was 78%). The book PDF goes from 530 to 457 pages with the same fonts, colours and structure. The A4 preset is unchanged.
- `tools/build_pdf.py` takes `--base`, `--lh`, `--gap`, `--mx`, `--top`, `--bottom` and `--img` to try other densities without editing the code.
- Fixed a sentence break in section 7.5.

## Second edition, update 1 (November 2026)

**New**
- Section 18.4 now builds RL from the ground up before PPO: MDPs and value functions, the policy-gradient derivation, REINFORCE, baselines (with a variance experiment in `code/ch18_check.py`), reward-to-go and actor–critic, importance sampling over long sequences, trust regions, and why LLMs use policy gradients rather than Q-learning.
- Appendix F: a 40-question diagnostic with an answer key and a rule for turning scores into a study plan.
- Chapter 37: answers collapsed for active recall on GitHub; section G (six follow-up chains, four levels deep) and section H (weak, acceptable and strong answers to the same question); ten new questions (241–250).
- `flashcards/`: 695 question–answer cards for Anki or any spaced-repetition app, generated by `tools/export_flashcards.py`.
- Research from the second half of 2025: GSPO and CISPO, the training–inference mismatch and truncated importance sampling, batch-invariant kernels, on-policy distillation, rubrics as rewards and RL scaling curves (Chapter 19, new section 19.12); MuonClip and mHC (Chapter 7); DeepSeek Sparse Attention (Chapter 11); nine new entries in the paper map (Appendix D).
- Two-week and seventy-two-hour refresh plans in the Overview.
- Continuous integration (`.github/workflows/check.yml`): Markdown and link checks (`tools/check_book.py`), flashcards up to date, and every chapter script run (`tools/run_code.py`).

**Fixed**
- `code/ch19_check.py` crashed on a `%` in a format string before printing its last three results.
- Section 6.9 repeated the 5.7 debugging checklist and had a broken numbered list; it now covers PyTorch-specific bugs only.
- `code/requirements.txt` pins the versions the scripts were run with.
- All 102 figures had the alt text "Diagram"; each now uses the first sentence of its caption.
