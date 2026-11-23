# Writing guide: from checklist to textbook

The handbook is for data scientists and ML engineers preparing for research and applied roles. Most readers have met every topic before; what they lack is the *why*, at the depth an interviewer probes. Write each section so that someone who half-remembers a concept finishes it able to explain it, derive it and use it.

Chapter 1, section 1.6 (entropy, cross-entropy and KL divergence), is the reference for the target depth.

## Chapter openings

- The italic intro (the "Part … ·" paragraph) stays short: where the chapter sits and what it assumes.
- **Why it matters** is two to four paragraphs, not one. Say what problem the chapter's ideas solve, how they connect to each other, the order the chapter takes and why, and what an interviewer is really testing.
- **Key concepts** stays a bulleted list: it is the map, not the explanation.

## Numbered sections

Each numbered section opens with one or two paragraphs that set up the question it answers, before any definition, formula or table. A reader should know why they are reading the section before they meet the first symbol.

Then, for each core concept:

1. **Motivation.** What breaks, or what question arises, without it.
2. **Intuition.** A plain-language reading, an analogy or a picture, before the formula.
3. **Definition.** The formula as a `$$ … $$` display block when it is central, with every symbol explained in the sentence after it.
4. **Worked example.** Small, concrete numbers that the reader can check by hand, computed and verified (keep the existing numbers where they exist; recompute any you add).
5. **Properties and consequences.** Each with a sentence on why it matters in practice.
6. **Where it shows up.** Each use with the reason it needs this concept, and a chapter cross-reference.

Use `###` subheadings inside long sections so the structure is visible at a glance. Prefer prose to bullets for explanations; keep bullets for genuine lists. Keep tables for comparisons and worked numbers.

## What not to change

- Do not remove content: expand around it. Existing worked numbers, figures, code and cross-references stay.
- Do not change the end-of-chapter blocks (Pitfalls, Interview questions, Practice, Read next, Done when) beyond fixing references. The question bank in chapter 37 feeds the flashcard export; do not use `$$` blocks there.
- Keep British spelling, the book's tone (direct, concrete, no hype) and section numbering, so cross-references stay valid.
- Every number in the text must be checked: compute it, do not estimate it.

## Checks

```
python3 tools/check_book.py               # 0 problem(s)
python3 tools/export_flashcards.py --check
```
