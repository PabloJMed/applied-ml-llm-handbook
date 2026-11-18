"""Build Applied-ML-and-LLM-Handbook.pdf from the Markdown chapters.

    pip install -r tools/requirements.txt
    python tools/build_pdf.py            # writes Applied-ML-and-LLM-Handbook.pdf in the repo root

The Markdown in chapters/ is the single source. Layout: A4, compact typography
(Inter body, Source Serif 4 headings, JetBrains Mono code; all SIL OFL, in tools/fonts/),
a linked contents page, PDF bookmarks, running headers and page numbers.
"""
from __future__ import annotations

import datetime as dt
import html
import re
import sys
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
FONTS = ROOT / "tools" / "fonts"
OUT = ROOT / "Applied-ML-and-LLM-Handbook.pdf"

TITLE = "Applied ML & LLM Handbook"
SUBTITLE = ("The theory, the code, the system design and the interview, in one place: "
            "a self-study guide for data scientists and ML engineers preparing for "
            "senior applied ML and LLM roles.")
AUTHOR = "Pablo Medina"
CREDIT = "Written with the guidance of Claude Opus 5.5 (Anthropic)"
EDITION = "Edition of November 2026"

PARTS = [
    ("I", "Foundations", ["01", "02", "03", "04", "05"]),
    ("II", "Deep learning", ["06", "07", "08"]),
    ("III", "NLP and transformers", ["09", "10", "11", "12"]),
    ("IV", "Large language models", ["13", "13a", "14", "15", "16"]),
    ("V", "Retrieval and RAG", ["17", "18", "19"]),
    ("VI", "Agents", ["20", "21"]),
    ("VII", "Evaluating LLM systems", ["22", "23", "24"]),
    ("VIII", "Recommender and search systems", ["25", "26", "27"]),
    ("IX", "ML systems and production", ["28", "29", "30"]),
    ("X", "Interview craft", ["31", "32", "33", "34"]),
    ("", "Appendices", ["A", "B", "C", "D", "E"]),
]

# Page geometry (points). A4 with modest margins.
PAGE = pymupdf.paper_rect("a4")
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 46, 50, 46
ACCENT = (0.757, 0.439, 0.310)      # terracotta, as in the first edition
INK = (0.13, 0.13, 0.12)
MUTED = (0.45, 0.44, 0.42)
COVER_BG = (0.973, 0.961, 0.937)

CSS = """
@font-face { font-family: sans; src: url(Inter-Regular.ttf); }
@font-face { font-family: sans; src: url(Inter-Italic.ttf); font-style: italic; }
@font-face { font-family: sans; src: url(Inter-SemiBold.ttf); font-weight: bold; }
@font-face { font-family: sans; src: url(Inter-SemiBoldItalic.ttf); font-weight: bold; font-style: italic; }
@font-face { font-family: serif; src: url(SourceSerif4-Regular.ttf); }
@font-face { font-family: serif; src: url(SourceSerif4-It.ttf); font-style: italic; }
@font-face { font-family: serif; src: url(SourceSerif4-Semibold.ttf); font-weight: bold; }
@font-face { font-family: mono; src: url(JetBrainsMono-Regular.ttf); }
@font-face { font-family: mono; src: url(JetBrainsMono-Bold.ttf); font-weight: bold; }
@font-face { font-family: mono; src: url(JetBrainsMono-Italic.ttf); font-style: italic; }

body { margin: 0; padding: 0; font-family: sans; font-size: 9.4pt; line-height: 1.36; color: #222220; }
p { margin: 0 0 4.2pt 0; text-align: left; }
h1, h2, h3, h4 { font-family: serif; font-weight: bold; color: #222220; }
h1 { font-size: 21pt; line-height: 1.12; margin: 0 0 9pt 0; }
h2 { font-size: 13pt; line-height: 1.2; margin: 9pt 0 3pt 0; }
h3 { font-size: 10.6pt; line-height: 1.25; margin: 7pt 0 2pt 0; }
h4 { font-size: 9.6pt; margin: 6pt 0 2pt 0; }
a { color: #2d5f9a; text-decoration: none; }
strong, b { font-weight: bold; }
code { font-family: mono; font-size: 8.2pt; }
pre { font-family: mono; font-size: 7.5pt; line-height: 1.3; background-color: #f3f1ec;
      padding: 0 6pt; margin: 3pt 0 6pt 0; white-space: pre-wrap; }
pre code { font-size: 7.5pt; }
ul, ol { margin: 0 0 4.2pt 0; padding-left: 13pt; }
li { margin: 0 0 1.4pt 0; }
li p { margin: 0 0 1.4pt 0; }
blockquote { margin: 3pt 0 5pt 0; padding-left: 8pt; border-left: 1.5pt solid #c1704f; color: #55534f; }
table { border-collapse: collapse; margin: 3pt 0 6pt 0; font-size: 8.2pt; line-height: 1.28; width: 100%; }
th { font-weight: bold; text-align: left; border-bottom: 0.9pt solid #b9b2a5; }
th, td { border: 0.5pt solid #d9d5cc; padding: 2.2pt 4pt; vertical-align: top; }
td code, th code { font-size: 7.4pt; }
img { width: 82%; }
p.fig { text-align: center; margin: 4pt 0 6pt 0; }
hr { border: none; border-top: 0.5pt solid #d9d5cc; margin: 6pt 0; }

.chapter { page-break-before: always; }
.partlabel { font-family: sans; font-weight: bold; font-size: 8pt; color: #c1704f; margin: 0 0 2pt 0; }
.partintro { font-size: 8.6pt; color: #6b6964; margin: 0 0 10pt 0; }

.cover { page-break-after: always; }
.cover-title { font-family: sans; font-size: 40pt; line-height: 1.06; color: #222220; margin: 150pt 0 14pt 0; }
.cover-sub { font-family: sans; font-size: 12.5pt; line-height: 1.4; color: #55534f; margin: 22pt 60pt 0 0; }
.cover-author { font-family: serif; font-weight: bold; font-size: 17pt; color: #222220; margin: 34pt 0 2pt 0; }
.cover-credit { font-family: sans; font-size: 10pt; color: #6b6964; margin: 0; }
.cover-meta { font-family: sans; font-weight: bold; font-size: 8pt; color: #c1704f; margin: 54pt 0 5pt 0; }
.cover-parts { font-size: 9pt; color: #55534f; line-height: 1.55; }
.cover-edition { font-size: 8.5pt; color: #55534f; margin: 50pt 0 0 0; }

.toc h1 { margin-bottom: 6pt; }
.toc-part { font-family: sans; font-weight: bold; font-size: 7.6pt; color: #c1704f; margin: 6pt 0 1pt 0; }
.toc table { margin: 0; font-size: 9pt; border-collapse: collapse; }
.toc td { border: none; padding: 0.6pt 0; }
.toc td.num { width: 26pt; color: #c1704f; }
.toc td.pg { width: 30pt; text-align: right; color: #6b6964; }
.colophon { font-size: 7.8pt; color: #6b6964; margin-top: 10pt; }
"""


def chapter_files() -> dict[str, Path]:
    files = {}
    for p in sorted(CHAPTERS.glob("*.md")):
        key = p.name.split("-", 1)[0]
        files[key] = p
    return files


def make_md() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "typographer": False}).enable(["table", "strikethrough"])
    return md


def chapter_html(md: MarkdownIt, key: str, path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    title = first.lstrip("# ").strip()
    body = md.render(rest)
    # The first paragraph repeats the part name ("Part I · Foundations. Assumes ..."): the part is
    # already shown above the title, so drop it and set the rest as a short muted intro.
    body = re.sub(r"^<p>Part [IVX]+ · (?:[^.<:]*\.\s*)?", '<p class="partintro">', body, count=1)
    # Images: keep them on their own centred line.
    body = re.sub(r"<p><img ([^>]*)></p>", r'<p class="fig"><img \1></p>', body)
    return title, body


def build_html(positions: list | None) -> str:
    files = chapter_files()
    md = make_md()
    pages = {}
    if positions:
        for pos in positions:
            if pos.heading == 1 and pos.id and pos.open_close & 1:
                pages.setdefault(pos.id, pos.page_num)

    def pg(anchor: str) -> str:
        n = pages.get(anchor)
        return str(n) if n is not None else "000"

    out = []
    # ---- cover
    parts_list = "<br>".join(
        (f"{num} · {name}" if num else name) for num, name, _ in PARTS)
    n_chapters = sum(len(keys) for num, _, keys in PARTS if num)
    out.append(f"""
<div class="cover">
<p class="cover-title">Applied ML<br>&amp; LLM Handbook</p>
<p class="cover-sub">{html.escape(SUBTITLE)}</p>
<p class="cover-author">{AUTHOR}</p>
<p class="cover-credit">{html.escape(CREDIT)}</p>
<p class="cover-meta">{n_chapters} CHAPTERS · {len(PARTS) - 1} PARTS · 5 APPENDICES</p>
<p class="cover-parts">{parts_list}</p>
<p class="cover-edition">{EDITION}</p>
</div>""")

    # ---- contents
    toc = ['<div class="toc"><h1 id="contents">Contents</h1><table>',
           f'<tr><td class="num"></td><td><a href="#ch-00">Overview: how to use this handbook</a></td><td class="pg">{pg("ch-00")}</td></tr></table>']
    for num, name, keys in PARTS:
        label = f"Part {num} · {name}" if num else name
        toc.append(f'<p class="toc-part">{label.upper()}</p><table>')
        for key in keys:
            title, _ = chapter_html(md, key, files[key])
            ttl = title.split(" · ", 1)[-1]
            shown = key.lstrip("0") if key[0].isdigit() else key
            toc.append(f'<tr><td class="num">{shown}</td><td><a href="#ch-{key}">{html.escape(ttl)}</a></td>'
                       f'<td class="pg">{pg("ch-" + key)}</td></tr>')
        toc.append("</table>")
    toc.append(f'<p class="colophon">{AUTHOR}. {html.escape(CREDIT)}. Text, diagrams and this PDF: '
               'CC BY-NC-SA 4.0; code: MIT. Source, code and corrections: '
               '<a href="https://github.com/PabloJMed/applied-ml-llm-handbook">'
               'github.com/PabloJMed/applied-ml-llm-handbook</a>.</p></div>')
    out.append("".join(toc))

    # ---- chapters
    def add_chapter(key: str, part_label: str | None):
        title, body = chapter_html(md, key, files[key])
        head = f'<p class="partlabel">{part_label.upper()}</p>' if part_label else ""
        out.append(f'<div class="chapter">{head}<h1 id="ch-{key}">{html.escape(title)}</h1>{body}</div>')

    add_chapter("00", None)
    for num, name, keys in PARTS:
        label = f"Part {num} · {name}" if num else name
        for i, key in enumerate(keys):
            add_chapter(key, label if i == 0 else None)
    return "\n".join(out)


def rectfn(rect_num, filled):
    where = pymupdf.Rect(MARGIN_X, MARGIN_TOP, PAGE.width - MARGIN_X, PAGE.height - MARGIN_BOTTOM)
    return PAGE, where, None


def main() -> int:
    archive = pymupdf.Archive()
    archive.add(pymupdf.Archive(str(FONTS)))
    archive.add(pymupdf.Archive(str(CHAPTERS)))

    collected: list = []

    def contentfn(positions):
        collected[:] = positions
        return build_html(positions)

    doc = pymupdf.Story.write_stabilized_with_links(
        contentfn, rectfn, user_css=CSS, em=9.4, archive=archive, add_header_ids=False)

    # positions -> chapter start pages (1-based page numbers from the stabilised layout)
    starts = {}
    sections = []
    for pos in collected:
        if not (pos.open_close & 1) or not pos.heading:
            continue
        if pos.heading == 1 and pos.id:
            starts.setdefault(pos.id, pos.page_num)
        if pos.heading in (1, 2) and pos.text:
            sections.append((pos.heading, pos.text.strip(), pos.page_num, pos.id))

    files = chapter_files()
    md = make_md()
    titles = {k: chapter_html(md, k, p)[0] for k, p in files.items()}

    # ---- running headers, page numbers, cover decoration
    order = ["00"] + [k for _, _, keys in PARTS for k in keys]
    chapter_of_page = {}
    for k in order:
        s = starts.get(f"ch-{k}")
        if s:
            chapter_of_page[s] = k
    cur = None
    font_sans = pymupdf.Font(fontfile=str(FONTS / "Inter-Regular.ttf"))
    for pno in range(doc.page_count):
        page = doc[pno]
        page.wrap_contents()   # the Story output leaves a flipped CTM; isolate it before adding text
        n = pno + 1
        if n == 1:   # cover
            shape = page.new_shape()
            shape.draw_rect(page.rect); shape.finish(fill=COVER_BG, color=None)
            shape.draw_rect(pymupdf.Rect(0, 0, 18, page.rect.height)); shape.finish(fill=ACCENT, color=None)
            shape.commit(overlay=False)
            hits = page.search_for("LLM Handbook")
            y = (hits[0].y1 + 12) if hits else 260
            page.draw_line((MARGIN_X, y), (MARGIN_X + 92, y), color=ACCENT, width=1.6)
            continue
        if n in chapter_of_page:
            cur = chapter_of_page[n]
        tw = pymupdf.TextWriter(page.rect)
        label = str(n)
        tw.append((page.rect.width / 2 - font_sans.text_length(label, 7.5) / 2, page.rect.height - 22),
                  label, font=font_sans, fontsize=7.5)
        if cur and n not in chapter_of_page:
            tw.append((MARGIN_X, 30), titles[cur], font=font_sans, fontsize=6.8)
        tw.write_text(page, color=MUTED)

    # ---- bookmarks
    toc = [[1, "Cover", 1], [1, "Contents", 2]]
    key_by_id = {f"ch-{k}": k for k in order}
    part_of = {keys[0]: (f"Part {num} · {name}" if num else name) for num, name, keys in PARTS}
    in_part = False
    for level, text, page, anchor in sections:
        if level == 1 and anchor in key_by_id:
            k = key_by_id[anchor]
            if k in part_of:
                toc.append([1, part_of[k], page]); in_part = True
            toc.append([2 if in_part else 1, titles[k], page])
        elif level == 2 and len(toc) > 2:
            toc.append([toc[-1][0] + 1 if toc[-1][0] < 3 and toc[-1][1] in titles.values() else toc[-1][0], text, page])
    # normalise: level-2 sections sit one below their chapter
    fixed, chap_level = [], 1
    for lvl, text, page in toc:
        if text in titles.values():
            chap_level = lvl; fixed.append([lvl, text, page])
        elif text.startswith("Part ") or text == "Appendices" or text in ("Cover", "Contents"):
            fixed.append([1, text, page])
        else:
            fixed.append([chap_level + 1, text, page])
    doc.set_toc(fixed)

    doc.set_metadata({
        "title": TITLE,
        "author": AUTHOR,
        "subject": "A self-study guide for senior applied ML and LLM roles",
        "keywords": "machine learning, LLM, RAG, MLOps, system design, interview preparation",
        "creator": f"{AUTHOR}, written with the guidance of Claude Opus 5.5",
        "producer": f"tools/build_pdf.py, PyMuPDF {pymupdf.VersionBind}",
        "creationDate": pymupdf.get_pdf_now(),
        "modDate": pymupdf.get_pdf_now(),
    })
    doc.subset_fonts()
    doc.save(OUT, garbage=4, deflate=True, deflate_fonts=True, deflate_images=True)
    print(f"wrote {OUT.relative_to(ROOT)}: {doc.page_count} pages, {OUT.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
