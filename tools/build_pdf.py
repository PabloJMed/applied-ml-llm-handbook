"""Build AI-and-LLM-Scientist-Handbook.pdf from the Markdown chapters.

    pip install -r tools/requirements.txt
    python tools/build_pdf.py              # book trim (7 x 10 in), the default
    python tools/build_pdf.py --page a4    # A4 for printing

The Markdown in chapters/ is the single source. Typography: Source Serif 4 for body text,
Inter for headings, tables and labels, JetBrains Mono for code (all SIL OFL, in tools/fonts/).
The PDF has a linked contents page, bookmarks, running headers and page numbers.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pymupdf
from fontTools.ttLib import TTFont
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
FONTS = ROOT / "tools" / "fonts"
OUT = ROOT / "AI-and-LLM-Scientist-Handbook.pdf"

TITLE = "The AI & LLM Scientist Handbook"
SUBTITLE = "A self-study guide for research scientists and ML engineers working with large language models"
AUTHORS = "Pablo Medina & Claude Opus 5.5"
EDITION = "Second edition · November 2026"
REPO = "github.com/PabloJMed/applied-ml-llm-handbook"

PARTS = [
    ("I", "Foundations", ["01", "02", "03", "04"]),
    ("II", "Deep learning and training dynamics", ["05", "06", "07", "08"]),
    ("III", "Transformers and architectures", ["09", "10", "11", "12", "13", "14"]),
    ("IV", "Pretraining", ["15", "16", "17"]),
    ("V", "Post-training", ["18", "19", "20"]),
    ("VI", "Inference and prompting", ["21", "22"]),
    ("VII", "Evaluation science", ["23", "24", "25"]),
    ("VIII", "Interpretability, alignment and safety", ["26", "27", "28"]),
    ("IX", "Retrieval and agents", ["29", "30", "31", "32", "33"]),
    ("X", "Research craft and interviews", ["34", "35", "36", "37", "38", "39"]),
    ("XI", "Applied ML systems (optional track)", ["40", "41", "42", "43", "44", "45", "46", "47", "48"]),
    ("", "Appendices", ["A", "B", "C", "D", "E"]),
]

# Page presets: page size, margins (left/right, top, bottom) and base font size.
PRESETS = {
    "book": dict(page=pymupdf.Rect(0, 0, 504, 720), mx=48, top=58, bottom=54, base=9.0),
    "a4": dict(page=pymupdf.paper_rect("a4"), mx=66, top=70, bottom=62, base=9.8),
}

ACCENT = (0.722, 0.384, 0.247)      # terracotta
INK = (0.12, 0.12, 0.11)
MUTED = (0.47, 0.46, 0.44)
RULE = (0.82, 0.80, 0.76)
COVER_BG = (0.973, 0.961, 0.937)


def css(base: float) -> str:
    s = lambda k: f"{base * k:.2f}pt"          # sizes relative to the body size
    return f"""
@font-face {{ font-family: sans; src: url(Inter-Regular.ttf); }}
@font-face {{ font-family: sans; src: url(Inter-Italic.ttf); font-style: italic; }}
@font-face {{ font-family: sans; src: url(Inter-SemiBold.ttf); font-weight: bold; }}
@font-face {{ font-family: sans; src: url(Inter-SemiBoldItalic.ttf); font-weight: bold; font-style: italic; }}
@font-face {{ font-family: serif; src: url(SourceSerif4-Regular.ttf); }}
@font-face {{ font-family: serif; src: url(SourceSerif4-It.ttf); font-style: italic; }}
@font-face {{ font-family: serif; src: url(SourceSerif4-Semibold.ttf); font-weight: bold; }}
@font-face {{ font-family: serif; src: url(SourceSerif4-SemiboldIt.ttf); font-weight: bold; font-style: italic; }}
@font-face {{ font-family: mono; src: url(JetBrainsMono-Regular.ttf); }}
@font-face {{ font-family: mono; src: url(JetBrainsMono-Bold.ttf); font-weight: bold; }}
@font-face {{ font-family: mono; src: url(JetBrainsMono-Italic.ttf); font-style: italic; }}

body {{ margin: 0; padding: 0; font-family: serif; font-size: {s(1)}; line-height: 1.42; color: #1f1f1d; }}
p {{ margin: 0 0 {s(0.55)} 0; text-align: left; }}
strong, b {{ font-weight: bold; color: #141413; }}
em, i {{ font-style: italic; }}
a {{ color: #8f4a2c !important; text-decoration: none !important; }}

h1, h2, h3, h4 {{ font-family: sans; font-weight: bold; color: #1a1a18; }}
h2 {{ font-size: {s(1.32)}; line-height: 1.25; margin: {s(1.45)} 0 {s(0.5)} 0; }}
h3 {{ font-size: {s(1.08)}; line-height: 1.3; margin: {s(1.15)} 0 {s(0.35)} 0; }}
h4 {{ font-size: {s(1.0)}; margin: {s(1.1)} 0 {s(0.3)} 0; }}
span.sec {{ color: #b8623f; }}

code {{ font-family: mono; font-size: {s(0.82)}; color: #3b3a36; }}
pre {{ font-family: mono; font-size: {s(0.72)}; line-height: 1.45; color: #2b2a27; background-color: #f5f2ec;
      border-top: {s(0.55)} solid #f5f2ec; border-bottom: {s(0.55)} solid #f5f2ec;
      padding: 0 {s(0.7)}; margin: {s(0.35)} 0 {s(1.0)} 0; white-space: pre-wrap; }}
pre code {{ font-size: {s(0.72)}; color: #2b2a27; }}

ul, ol {{ margin: 0 0 {s(0.7)} 0; padding-left: {s(1.6)}; }}
li {{ margin: 0 0 {s(0.22)} 0; }}
li p {{ margin: 0 0 {s(0.25)} 0; }}
blockquote {{ margin: {s(0.4)} 0 {s(0.8)} 0; padding-left: {s(0.9)}; border-left: 1.2pt solid #c98a6c; color: #4f4d48; }}

table {{ border-collapse: collapse; margin: {s(0.45)} 0 {s(1.05)} 0; font-family: sans; font-size: {s(0.8)};
        line-height: 1.38; width: 100%; border-top: 0.9pt solid #3a3935; border-bottom: 0.9pt solid #3a3935; }}
th {{ font-weight: bold; text-align: left; border-bottom: 0.6pt solid #3a3935; padding: {s(0.4)} {s(0.6)}; vertical-align: bottom; }}
td {{ border-bottom: 0.35pt solid #dcd7cd; padding: {s(0.36)} {s(0.6)}; vertical-align: top; }}
td code, th code {{ font-size: {s(0.74)}; }}
td strong {{ font-family: sans; }}

img {{ width: 78%; }}
p.fig {{ text-align: center; margin: {s(0.5)} 0 {s(0.25)} 0; }}
p.caption {{ font-family: sans; font-size: {s(0.78)}; color: #6e6c66; text-align: center; margin: 0 0 {s(1.0)} 0; }}
hr {{ border: none; border-top: 0.5pt solid #d8d2c6; margin: {s(0.8)} 0; }}

.chapter {{ page-break-before: always; }}
.brk {{ page-break-before: always; }}
p.spacer {{ margin: 0; font-size: 1pt; line-height: 1; }}
.tight p, .tight li {{ line-height: 1.36; }}
.tight p {{ margin-bottom: {s(0.45)}; }}
.tighter p, .tighter li {{ line-height: 1.31; }}
.tighter p {{ margin-bottom: {s(0.38)}; }}
.tighter h2 {{ margin-top: {s(1.2)}; }}
.partlabel {{ font-family: sans; font-weight: bold; font-size: {s(0.72)}; color: #b8623f; margin: {s(1.0)} 0 {s(1.6)} 0; }}
.chapnum {{ font-family: sans; font-size: {s(0.95)}; color: #b8623f; margin: 0 0 {s(0.25)} 0; }}
h1 {{ font-size: {s(2.25)}; line-height: 1.12; margin: 0 0 {s(0.9)} 0; padding-bottom: {s(0.9)};
     border-bottom: 0.6pt solid #d8d2c6; }}
.partintro {{ font-style: italic; color: #5d5b56; margin: 0 0 {s(1.4)} 0; }}

.box {{ border-left: 1.6pt solid #d9a184; padding: 0 0 0 {s(1.0)}; margin: {s(0.4)} 0 {s(1.2)} 0; }}
.box h2 {{ font-size: {s(0.86)}; color: #b8623f; margin: 0 0 {s(0.45)} 0; }}
.box ul {{ margin-bottom: 0; }}
.box p {{ margin-bottom: {s(0.45)}; }}

.cover {{ page-break-after: always; }}
.dedication {{ page-break-after: always; }}
.dedication p {{ font-family: serif; font-style: italic; font-size: {s(1.15)}; line-height: 1.5; color: #3d3b37; text-align: center; margin: 0; }}
.dedication p.top {{ padding-top: {s(16)}; }}
.dedication p.sig {{ font-style: normal; font-family: sans; font-size: {s(0.82)}; color: #8a8882; margin: {s(0.6)} 0 {s(3.2)} 0; }}
.cover-title {{ font-family: sans; font-weight: bold; font-size: {s(3.6)}; line-height: 1.08; color: #1a1a18; margin: {s(11)} 0 0 0; }}
.cover-sub {{ font-family: serif; font-size: {s(1.22)}; line-height: 1.45; color: #4f4d48; margin: {s(2.6)} {s(2)} 0 0; }}
.cover-author {{ font-family: sans; font-weight: bold; font-size: {s(1.45)}; color: #1a1a18; margin: {s(3.2)} 0 0 0; }}
.cover-meta {{ font-family: sans; font-weight: bold; font-size: {s(0.74)}; color: #b8623f; margin: {s(4.2)} 0 {s(0.5)} 0; }}
.cover-parts {{ font-family: sans; font-size: {s(0.86)}; color: #55534e; line-height: 1.6; }}
.cover-edition {{ font-family: sans; font-size: {s(0.8)}; color: #6e6c66; margin: {s(3.4)} 0 0 0; }}

.toc h1 {{ border-bottom: none; padding-bottom: 0; margin-bottom: {s(0.6)}; }}
.toc-part {{ font-family: sans; font-weight: bold; font-size: {s(0.7)}; color: #b8623f; margin: {s(1.05)} 0 {s(0.15)} 0; }}
.toc table {{ margin: 0; font-family: sans; font-size: {s(0.88)}; line-height: 1.32; border: none; }}
.toc td {{ border: none; padding: {s(0.1)} 0; }}
.toc td.num {{ width: {s(2.4)}; color: #b8623f; }}
.toc td.pg {{ width: {s(2.6)}; text-align: right; color: #6e6c66; }}
.toc a {{ color: #1f1f1d !important; }}
.colophon {{ font-family: sans; font-size: {s(0.74)}; color: #6e6c66; margin-top: {s(1.6)}; }}
"""


def prepare_fonts() -> Path:
    """Copy the fonts to a temporary folder, disabling Source Serif's ligatures.

    Its "ft" ligature has no Unicode mapping, so text extracted from the PDF (search, copy-paste)
    would read "soƤmax" for "softmax". The repository's font files are left unmodified.
    """
    tmp = Path(tempfile.mkdtemp(prefix="handbook-fonts-"))
    for src in FONTS.glob("*.ttf"):
        dst = tmp / src.name
        if src.name.startswith("SourceSerif4"):
            font = TTFont(src)
            if "GSUB" in font:
                for rec in font["GSUB"].table.FeatureList.FeatureRecord:
                    if rec.FeatureTag in ("liga", "clig", "dlig", "hlig"):
                        rec.Feature.LookupListIndex = []
                        rec.Feature.LookupCount = 0
            font.save(dst)
        else:
            shutil.copy(src, dst)
    return tmp


BREAKS: set[str] = set()      # element ids that must start a new page (filled during layout)
TALL: set[str] = set()        # tables, code blocks and boxes
TIGHTER: set[str] = set()     # second step for chapters the first step did not fix
TIGHT: set[str] = set()       # chapters whose last page held only a few lines: set slightly tighter
KEEP: set[str] = set()        # headings and lead-in lines that must stay on the page of the next block


def add_ids(key: str, body: str) -> str:
    """Give every block a stable id and mark the ones the previous layout pass found stranded at a
    page bottom with a page break."""
    counter = iter(range(1000000))
    state = {"prev": None, "end": 0}

    def tag(m):
        el, attrs = m.group(1), m.group(2) or ""
        eid = f"e{key}-{next(counter)}"
        prev, gap = state["prev"], body[state["end"]:m.start()]
        state["prev"], state["end"] = eid, m.end()
        if prev in BREAKS and (prev in KEEP or not gap.strip()):
            # the heading or lead-in before this block (or the box this block opens) already starts
            # a new page: no second break
            return f'<{el} id="{eid}"{attrs}>'
        if eid in BREAKS and el in ("table", "pre", "ul", "ol", "div"):
            return f'<p class="brk spacer">&#8203;</p><{el} id="{eid}"{attrs}>'
        if eid in BREAKS:
            if 'class="' in attrs:
                attrs = attrs.replace('class="', 'class="brk ', 1)
            else:
                attrs = ' class="brk"' + attrs
        return f'<{el} id="{eid}"{attrs}>'
    # pre-compute which ids will be headings or lead-in lines (same numbering as tag() below)
    n = 0
    for m in re.finditer(r'<(h2|h3|h4|table|pre|p|ul|ol|div)((?: [^>]*)?)>', body):
        eid = f"e{key}-{n}"; n += 1
        if m.group(1) in ("h2", "h3", "h4"):
            KEEP.add(eid)
        elif m.group(1) == "p" and re.match(r'(?:(?!</p>).)*:(?:</strong>|</em>|</b>)*</p>\s*<(?:ul|ol|pre|table|div)', body[m.end():], flags=re.S):
            KEEP.add(eid)
    body = re.sub(r'<(h2|h3|h4|table|pre|p|ul|ol|div)((?: [^>]*)?)>', tag, body)
    for m in re.finditer(r'<(?:table|pre) id="([^"]+)"|<div id="([^"]+)" class="(?:brk )?box"', body):
        TALL.add(m.group(1) or m.group(2))
    for m in re.finditer(r'<h[234] id="([^"]+)"', body):
        KEEP.add(m.group(1))
    for m in re.finditer(r'<p id="([^"]+)"[^>]*>(?:(?!</p>).)*:</p>\s*<(?:ul|ol|pre|table|div)', body, flags=re.S):
        KEEP.add(m.group(1))
    return body


def chapter_files() -> dict[str, Path]:
    return {p.name.split("-", 1)[0]: p for p in sorted(CHAPTERS.glob("*.md"))}


def make_md() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": False, "typographer": False}).enable(["table", "strikethrough"])


CODE_COLS = 100                # characters per code line; set from the page geometry in main()


def wrap_code_line(line: str, cols: int) -> list[str]:
    """Fit one code line into `cols` characters: move a trailing comment above the code, then break
    at spaces with a hanging indent so continuation lines are visibly continuations."""
    if len(line) <= cols:
        return [line]
    indent = len(line) - len(line.lstrip(" "))
    m = re.search(r"\s{2,}(#|--|//) ", line)
    if m and line[:m.start()].strip() and len(line[:m.start()].rstrip()) <= cols:
        comment = " " * indent + line[m.start():].strip()
        return wrap_code_line(comment, cols) + [line[:m.start()].rstrip()]
    hang = " " * min(indent + 4, cols // 2)
    out, rest = [], line
    while len(rest) > cols:
        cut = max(rest.rfind(", ", 0, cols - 1) + 1, rest.rfind(" ", indent + 1, cols))
        if cut <= indent + 1:
            cut = cols
        out.append(rest[:cut].rstrip())
        rest = hang + rest[cut:].lstrip()
    out.append(rest)
    return out


def wrap_code(markdown: str, cols: int) -> str:
    def fix(m):
        lines = m.group(2).split("\n")
        return m.group(1) + "\n".join(x for l in lines for x in wrap_code_line(l, cols)) + m.group(3)
    return re.sub(r"(^```[^\n]*\n)(.*?)(\n```)", fix, markdown, flags=re.S | re.M)


def wrap_box(body: str, heading: str) -> str:
    """Wrap an h2 section (up to the next h2) in a ruled box."""
    m = re.search(rf"<h2>{re.escape(heading)}</h2>(.*?)(?=<h2>|\Z)", body, flags=re.S)
    if not m:
        return body
    return body[:m.start()] + f'<div class="box"><h2>{heading}</h2>{m.group(1)}</div>' + body[m.end():]


def chapter_html(md: MarkdownIt, key: str, path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\n---\n\n\[[^\n]*\]\([^\n]*\)[^\n]*\s*$", "\n", text)   # drop the web navigation footer
    first, _, rest = text.partition("\n")
    title = first.lstrip("# ").strip()
    body = md.render(wrap_code(rest, CODE_COLS))
    # The first paragraph repeats the part name ("Part I · Foundations. Assumes ..."); the part is
    # shown above the title, so drop it and set the rest as the chapter's italic intro.
    body = re.sub(r"^<p>Part [IVX]+ · (?:[^.<:]*\.\s*)?", '<p class="partintro">', body, count=1)
    body = re.sub(r'<p class="partintro">\s*</p>', "", body)
    # Figures: centred, with an italic caption line right after them styled as a caption.
    body = re.sub(r"<p><img ([^>]*)></p>\s*<p><em>(.*?)</em></p>", r'<p class="fig"><img \1></p><p class="caption">\2</p>', body, flags=re.S)
    body = re.sub(r"<p><img ([^>]*)></p>", r'<p class="fig"><img \1></p>', body)
    # Section numbers in the accent colour.
    body = re.sub(r"<h([23])>([0-9]+[a-z]?\.[0-9]+(?:\.[0-9]+)?|[A-E]\.[0-9]+) ", r'<h\1><span class="sec">\2</span>&#160;&#160;', body)
    for h in ("Key concepts", "Done when"):
        body = wrap_box(body, h)
    return title, add_ids(key, body)


def build_html(positions: list | None) -> str:
    files = chapter_files()
    md = make_md()
    pages = {}
    for pos in positions or []:
        if pos.heading == 1 and pos.id and pos.open_close & 1:
            pages.setdefault(pos.id, pos.page_num)
    pg = lambda anchor: str(pages.get(anchor, "000"))

    out = []
    parts_list = "<br>".join((f"{num} · {name}" if num else name) for num, name, _ in PARTS)
    n_chapters = sum(len(keys) for num, _, keys in PARTS if num)
    out.append(f"""
<div class="cover">
<p class="cover-title">The AI &amp; LLM<br>Scientist Handbook</p>
<p class="cover-sub">{html.escape(SUBTITLE)}</p>
<p class="cover-author">{html.escape(AUTHORS)}</p>
<p class="cover-meta">{n_chapters} CHAPTERS · {len(PARTS) - 1} PARTS · 5 APPENDICES</p>
<p class="cover-parts">{parts_list}</p>
<p class="cover-edition">{EDITION}</p>
</div>""")
    out.append("""
<div class="dedication">
<p class="top">To whoever makes it past Chapter 13.<br>I see you.</p>
<p class="sig">— Pablo</p>
<p>To Claudia,<br>who would have done it in half the tokens.</p>
<p class="sig">— Opus</p>
</div>""")

    toc = ['<div class="toc"><h1 id="contents">Contents</h1><table>',
           f'<tr><td class="num"></td><td><a href="#ch-00">Overview: how to use this handbook</a></td><td class="pg">{pg("ch-00")}</td></tr></table>']
    for num, name, keys in PARTS:
        label = f"Part {num} · {name}" if num else name
        toc.append(f'<p class="toc-part">{html.escape(label.upper())}</p><table>')
        for key in keys:
            title, _ = chapter_html(md, key, files[key])
            shown = key.lstrip("0") if key[0].isdigit() else key
            toc.append(f'<tr><td class="num">{shown}</td><td><a href="#ch-{key}">{html.escape(title.split(" · ", 1)[-1])}</a></td>'
                       f'<td class="pg">{pg("ch-" + key)}</td></tr>')
        toc.append("</table>")
    toc.append(f'<p class="colophon">{html.escape(AUTHORS)}. {EDITION}. Text, diagrams and this PDF: CC BY-NC-SA 4.0; '
               f'code: MIT. Source, code and corrections: <a href="https://{REPO}">{REPO}</a>.</p></div>')
    out.append("".join(toc))

    def add_chapter(key: str, part_label: str | None):
        title, body = chapter_html(md, key, files[key])
        num, _, name = title.partition(" · ")
        if not name:
            num, name = "", title
        head = f'<p class="partlabel">{html.escape(part_label.upper())}</p>' if part_label else ""
        label = ("Appendix " if key.isalpha() else "Chapter ") + num if num else ""
        chapnum = f'<p class="chapnum">{label}</p>' if label else ""
        cls = "chapter tighter" if key in TIGHTER else ("chapter tight" if key in TIGHT else "chapter")
        out.append(f'<div class="{cls}">{head}{chapnum}<h1 id="ch-{key}">{html.escape(name)}</h1>{body}</div>')

    add_chapter("00", None)
    for num, name, keys in PARTS:
        label = f"Part {num} · {name}" if num else name
        for i, key in enumerate(keys):
            add_chapter(key, label if i == 0 else None)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", choices=sorted(PRESETS), default="book")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    P = PRESETS[args.page]
    PAGE, MX, TOP, BOTTOM, BASE = P["page"], P["mx"], P["top"], P["bottom"], P["base"]

    global CODE_COLS
    code_size = BASE * 0.72
    CODE_COLS = int((PAGE.width - 2 * MX - 2 * 0.7 * BASE) / (0.6 * code_size)) - 2
    font_dir = prepare_fonts()
    archive = pymupdf.Archive()
    archive.add(pymupdf.Archive(str(font_dir)))
    archive.add(pymupdf.Archive(str(CHAPTERS)))

    def rectfn(rect_num, filled):
        return PAGE, pymupdf.Rect(MX, TOP, PAGE.width - MX, PAGE.height - BOTTOM), None

    collected: list = []

    line = BASE * 1.42
    floor = PAGE.height - BOTTOM                 # bottom of the text area

    def contentfn(positions):
        collected[:] = positions
        starts = [p for p in (positions or []) if (p.open_close & 1) and p.id and p.id.startswith("e")]
        for prev, nxt in zip(starts, starts[1:]):
            # a heading or lead-in line whose next block starts on a later page moves with it
            if prev.id in KEEP and nxt.page_num != prev.page_num and prev.rect[1] > TOP + line:
                BREAKS.add(prev.id)
        for p in starts:
            # tables and code blocks need room for a few lines before the page ends
            if p.id not in KEEP and p.rect[1] > floor - 3.5 * line and p.rect[1] > TOP + line:
                if re.match(r"e[0-9A-E]+-", p.id) and p.id in TALL:
                    BREAKS.add(p.id)
        return build_html(positions)

    def layout():
        return pymupdf.Story.write_stabilized_with_links(
            contentfn, rectfn, user_css=css(BASE), em=BASE, archive=archive, add_header_ids=False)

    doc = layout()
    # Breaks only accumulate during a layout run, so a break added while earlier content sat
    # differently can outlive its reason and leave a half-empty page. Drop every break whose previous
    # page is less than two-thirds full and lay out again: the pass re-adds the ones still needed.
    def content_bottom(page) -> float:
        """Lowest point of the text and images actually laid out on a page (no headers exist yet)."""
        ys = [b[3] for b in page.get_text("blocks")]
        ys += [r.y1 for x in page.get_images() for r in page.get_image_rects(x[0])]
        return max(ys, default=TOP)

    def prune():
        nonlocal doc
        for _ in range(4):
            first = {}
            for p in collected:
                if (p.open_close & 1) and p.id in BREAKS:
                    first.setdefault(p.id, p.page_num)
            # page_num is 1-based, so the page before a break's page is doc[pg - 2]
            stale = {i for i, pg in first.items()
                     if pg > 1 and content_bottom(doc[pg - 2]) < floor - 0.34 * (floor - TOP)}
            if not stale:
                return
            BREAKS.difference_update(stale)
            doc = layout()

    prune()
    # A chapter whose last page holds only a few lines or a lone box is set with slightly tighter
    # line spacing (1.36 instead of 1.42), which is usually enough to pull those lines back.
    def sparse_endings() -> set[str]:
        chapter_start = {p.id[3:]: p.page_num for p in collected
                         if (p.open_close & 1) and p.heading == 1 and p.id and p.id.startswith("ch-")}
        ordered = sorted(chapter_start.items(), key=lambda kv: kv[1])
        return {key for (key, _), (_, nxt) in zip(ordered, ordered[1:])
                if (content_bottom(doc[nxt - 2]) - TOP) / (floor - TOP) < 0.3}

    for target in (TIGHT, TIGHTER):
        found = sparse_endings() - TIGHTER
        if not found:
            break
        target.update(found)
        doc = layout()
        prune()

    starts, sections = {}, []
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

    # ---- cover decoration, running headers, page numbers
    order = ["00"] + [k for _, _, keys in PARTS for k in keys]
    chapter_of_page = {starts[f"ch-{k}"]: k for k in order if f"ch-{k}" in starts}
    font = pymupdf.Font(fontfile=str(FONTS / "Inter-Regular.ttf"))
    small = BASE * 0.72
    cur = None
    for pno in range(doc.page_count):
        page = doc[pno]
        page.wrap_contents()          # Story output leaves a flipped CTM; isolate it before adding content
        n = pno + 1
        W, H = page.rect.width, page.rect.height
        if n == 2:
            continue                  # dedication page: no number
        if n == 1:
            shape = page.new_shape()
            shape.draw_rect(page.rect); shape.finish(fill=COVER_BG, color=None)
            shape.draw_rect(pymupdf.Rect(0, 0, 14, H)); shape.finish(fill=ACCENT, color=None)
            shape.commit(overlay=False)
            hits = page.search_for("Scientist Handbook")
            y = (hits[0].y1 + BASE * 1.4) if hits else H * 0.35
            page.draw_line((MX, y), (MX + 80, y), color=ACCENT, width=1.4)
            continue
        if n in chapter_of_page:
            cur = chapter_of_page[n]
        tw = pymupdf.TextWriter(page.rect)
        label = str(n)
        lw = font.text_length(label, small)
        x = (W - MX - lw) if n % 2 else MX                # page numbers in the outer corner
        tw.append((x, H - BOTTOM * 0.5), label, font=font, fontsize=small)
        if cur and n not in chapter_of_page and n > 3:
            head = titles[cur]
            hw = font.text_length(head, small)
            hx = (W - MX - hw) if n % 2 else MX
            tw.append((hx, TOP * 0.55), head, font=font, fontsize=small)
            page.draw_line((MX, TOP * 0.55 + small * 0.75), (W - MX, TOP * 0.55 + small * 0.75), color=RULE, width=0.4)
        tw.write_text(page, color=MUTED)

    # ---- bookmarks: parts > chapters > sections
    key_by_id = {f"ch-{k}": k for k in order}
    part_of = {keys[0]: (f"Part {num} · {name}" if num else name) for num, name, keys in PARTS}
    toc = [[1, "Cover", 1], [1, "Dedication", 2], [1, "Contents", 3]]
    chap_level = 1
    for level, text, page, anchor in sections:
        if level == 1 and anchor in key_by_id:
            k = key_by_id[anchor]
            if k in part_of:
                toc.append([1, part_of[k], page])
            chap_level = 1 if k == "00" else 2
            toc.append([chap_level, titles[k], page])
        elif level == 2 and len(toc) > 2 and text not in ("Key concepts", "Done when"):
            toc.append([chap_level + 1, text.replace(" ", " "), page])
    doc.set_toc(toc)

    doc.set_metadata({
        "title": TITLE,
        "author": AUTHORS,
        "subject": "A self-study handbook for AI and LLM research scientists",
        "keywords": "LLM, pretraining, post-training, RLHF, reasoning, scaling laws, evaluation, interpretability, alignment, machine learning",
        "creator": AUTHORS,
        "producer": f"tools/build_pdf.py, PyMuPDF {pymupdf.VersionBind}",
        "creationDate": pymupdf.get_pdf_now(),
        "modDate": pymupdf.get_pdf_now(),
    })
    doc.subset_fonts()
    doc.save(args.out, garbage=4, deflate=True, deflate_fonts=True, deflate_images=True)
    shutil.rmtree(font_dir, ignore_errors=True)
    print(f"tighter spacing: chapters {sorted(TIGHT - TIGHTER)} (step 1), {sorted(TIGHTER)} (step 2)")
    print(f"wrote {args.out.name} ({args.page}): {doc.page_count} pages, {args.out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
