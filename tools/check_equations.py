"""Check that display equations typeset with matplotlib mathtext, as tools/build_pdf.py needs.

    python tools/check_equations.py                     # every chapter
    python tools/check_equations.py chapters/19-*.md    # some files

For each $$ ... $$ block, renders the LaTeX the way build_pdf.py does and reports blocks that
mathtext cannot parse (unsupported commands such as \\begin{aligned} or \\text) or that are wider
than the page. Much faster than a full PDF build; run it after adding or editing equations.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pdf  # noqa: E402


def main(argv: list[str]) -> int:
    root = build_pdf.ROOT if hasattr(build_pdf, "ROOT") else Path(__file__).resolve().parent.parent
    files = [Path(a) for a in argv] or sorted((root / "chapters").glob("*.md"))
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="eqcheck-"))
    problems, total = [], 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, m in enumerate(build_pdf.EQ_RE.finditer(text), 1):
            total += 1
            latex = " ".join(m.group(1).split())
            try:
                build_pdf.render_equation(latex, "Eq. 99.99", tmp / f"{i}.png")
            except BaseException as e:  # SystemExit for too-wide, ValueError for parse errors
                line = text[: m.start()].count("\n") + 1
                msg = type(e).__name__ + ": " + " ".join(str(e).split())
                problems.append(f"{path.name}:{line}: {msg[:160]}")
    for p in problems:
        print(p)
    print(f"checked {total} equations: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
