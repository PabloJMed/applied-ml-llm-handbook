"""Export the handbook's questions and model answers as flashcards.

    python tools/export_flashcards.py

Writes flashcards/handbook-flashcards.txt, a tab-separated file that Anki imports directly
(File > Import; the header lines set the separator, the HTML setting and the tags column), and
that most other spaced-repetition apps and spreadsheets can read.

Sources:
- Chapter 37's question bank (every question, with its level as a tag);
- each chapter's "Interview questions" written as **N. Question?** answer;
- Appendix F's diagnostic, with the answer key on the back.

Questions without a written answer (plain numbered lists) are skipped. Re-run after editing
the chapters; CI checks the file is up to date.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
OUT = ROOT / "flashcards" / "handbook-flashcards.txt"


def clean(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)        # links -> text
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)                 # bold
    s = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", s)   # italics
    s = s.replace("`", "")
    return re.sub(r"\s+", " ", s).strip()


def bank_cards() -> list[tuple[str, str, str]]:
    text = (CHAPTERS / "37-ml-depth-question-bank.md").read_text(encoding="utf-8")
    cards, section = [], ""
    pattern = re.compile(
        r"^### (?P<sec>[^\n]+)$|^\*\*(?P<n>\d+)\. \[(?P<lv>[BIS])\] (?P<q>[^\n]+?)\*\*\n\n<details><summary>Answer</summary>\n\n(?P<a>.+?)\n\n</details>",
        flags=re.M | re.S)
    for m in pattern.finditer(text):
        if m.group("sec"):
            section = m.group("sec")
            continue
        chap = re.search(r"\(Chapters? ([^)]+)\)", section)
        level = {"B": "basic", "I": "intermediate", "S": "senior"}[m.group("lv")]
        tags = f"bank level::{level} " + ("ch::" + re.sub(r"\D+", "_", chap.group(1)).strip("_") if chap else "")
        cards.append((f"[Q{m.group('n')}] {clean(m.group('q'))}", clean(m.group("a")), tags.strip()))
    return cards


def chapter_cards() -> list[tuple[str, str, str]]:
    cards = []
    for path in sorted(CHAPTERS.glob("[0-9][0-9]-*.md")):
        key = path.name[:2]
        if key == "37":
            continue
        text = path.read_text(encoding="utf-8")
        if "## Interview questions" not in text:
            continue
        seg = text.split("## Interview questions", 1)[1].split("\n## ", 1)[0]
        for m in re.finditer(r"^\*\*(\d+)\. (.+?)\*\* (.+)$", seg, flags=re.M):
            cards.append((f"[Ch {int(key)}] {clean(m.group(2))}", clean(m.group(3)), f"chapter ch::{int(key)}"))
    return cards


def diagnostic_cards() -> list[tuple[str, str, str]]:
    text = (CHAPTERS / "F-diagnostic.md").read_text(encoding="utf-8")
    body, _, key = text.partition("## F.9 Answer key")
    qs = dict(re.findall(r"^\*\*(\d+)\.\*\* (.+)$", body, flags=re.M))
    ans = dict(re.findall(r"^\*\*(\d+)\.\*\* (.+)$", key, flags=re.M))
    return [(f"[Diagnostic {n}] {clean(qs[n])}", clean(ans[n]), "diagnostic") for n in qs if n in ans]


def main() -> int:
    cards = bank_cards() + chapter_cards() + diagnostic_cards()
    for front, back, _ in cards:
        if "\t" in front + back:
            raise SystemExit(f"tab inside a card: {front[:60]}")
    lines = ["#separator:tab", "#html:false", "#tags column:3",
             "#notetype:Basic", "#deck:AI and LLM Scientist Handbook"]
    lines += [f"{f}\t{b}\t{t}" for f, b, t in cards]
    OUT.parent.mkdir(exist_ok=True)
    new = "\n".join(lines) + "\n"
    check = "--check" in sys.argv
    if check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != new:
            print(f"{OUT.relative_to(ROOT)} is out of date: run python tools/export_flashcards.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date ({len(cards)} cards)")
        return 0
    OUT.write_text(new, encoding="utf-8")
    counts = {"bank": len(bank_cards()), "chapters": len(chapter_cards()), "diagnostic": len(diagnostic_cards())}
    print(f"wrote {OUT.relative_to(ROOT)}: {len(cards)} cards {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
