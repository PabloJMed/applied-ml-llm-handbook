"""Static checks on the Markdown sources, run in CI and before building the PDF.

    python tools/check_book.py

Checks:
- every ```python block in chapters/ parses with Python's ast module;
- every relative link (chapters, code, images, #anchors in other chapters) points to a file or
  heading that exists;
- every chapter file is listed in the README contents and in tools/build_pdf.py;
- every $$ display-equation block is opened and closed on lines of its own;
- every <details> block is closed (collapsible answers render on GitHub and are flattened in the PDF).

Exits non-zero and lists the problems if any check fails.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    out, seen = set(), {}
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r"#{1,6} (.*)", line)
        if m:
            s = slug(m.group(1))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    return out


def main() -> int:
    problems: list[str] = []
    md_files = sorted(CHAPTERS.glob("*.md")) + [ROOT / "README.md", ROOT / "code" / "README.md"]
    anchor_cache: dict[Path, set[str]] = {}

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)

        # Python blocks parse
        for i, m in enumerate(re.finditer(r"^```python\n(.*?)^```", text, flags=re.S | re.M)):
            try:
                ast.parse(m.group(1))
            except SyntaxError as e:
                line = text[:m.start()].count("\n") + 1 + (e.lineno or 0)
                problems.append(f"{rel}:{line}: Python block does not parse: {e.msg}")

        # <details> blocks balanced
        if text.count("<details>") != text.count("</details>"):
            problems.append(f"{rel}: {text.count('<details>')} <details> but {text.count('</details>')} </details>")

        # $$ display equations: delimiters alone on their lines, in pairs, outside code blocks
        no_code = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
        if len(re.findall(r"^\$\$[ \t]*$", no_code, flags=re.M)) % 2:
            problems.append(f"{rel}: unbalanced $$ equation delimiters")

        # relative links and images
        prose = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
        for m in re.finditer(r"\]\(([^)\s]+)\)", prose):
            target = m.group(1)
            if re.match(r"[a-z]+:", target):
                continue                                  # http(s), mailto, ...
            file_part, _, anchor = target.partition("#")
            dest = (path.parent / file_part).resolve() if file_part else path
            if not dest.exists():
                problems.append(f"{rel}: broken link {target}")
                continue
            if anchor and dest.suffix == ".md":
                if dest not in anchor_cache:
                    anchor_cache[dest] = anchors(dest)
                if anchor not in anchor_cache[dest]:
                    problems.append(f"{rel}: missing anchor {target}")

    # every chapter is in the README contents and in the PDF build
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    build = (ROOT / "tools" / "build_pdf.py").read_text(encoding="utf-8")
    for path in sorted(CHAPTERS.glob("*.md")):
        key = path.name.split("-", 1)[0]
        if key != "00" and f"chapters/{path.name}" not in readme:
            problems.append(f"README.md: chapter {path.name} is not in the contents")
        if key != "00" and f'"{key}"' not in build:
            problems.append(f"tools/build_pdf.py: chapter key {key} is not in PARTS")

    for p in problems:
        print(p)
    print(f"checked {len(md_files)} Markdown files: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
