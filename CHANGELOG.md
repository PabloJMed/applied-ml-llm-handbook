# Changelog

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
