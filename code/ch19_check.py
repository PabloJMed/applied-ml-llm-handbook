import re

def n_tokens(text):
    return len(text.split())          # stand-in; use the model's tokenizer in practice

def split_markdown_sections(doc):
    """Split on headings, keeping the heading path as context for each section."""
    sections, path, buf = [], [], []
    for line in doc.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf:
                sections.append((" > ".join(path), "\n".join(buf).strip()))
                buf = []
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    if buf:
        sections.append((" > ".join(path), "\n".join(buf).strip()))
    return [(p, t) for p, t in sections if t]

def chunk_text(text, max_tokens=120, overlap=20):
    """Pack paragraphs (then sentences) into chunks up to max_tokens, with overlap."""
    units = []
    for para in [p for p in text.split("\n\n") if p.strip()]:
        if n_tokens(para) <= max_tokens:
            units.append(para.strip())
        else:
            units += [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
    chunks, current = [], []
    for unit in units:
        if current and n_tokens(" ".join(current + [unit])) > max_tokens:
            chunks.append(" ".join(current))
            tail = " ".join(current).split()[-overlap:] if overlap else []
            current = [" ".join(tail)] if tail else []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_document(doc, doc_id, title, max_tokens=120, overlap=20):
    out = []
    for path, body in split_markdown_sections(doc):
        for i, chunk in enumerate(chunk_text(body, max_tokens, overlap)):
            header = f"{title} > {path}" if path else title
            out.append({"id": f"{doc_id}#{len(out)}", "doc_id": doc_id, "section": path,
                        "text": f"[{header}]\n{chunk}"})
    return out

doc = """# Refunds
## Late orders
If your order arrives more than 30 minutes after the promised time, you can request a voucher.
Vouchers are worth 20% of the order value, up to 10 euros.

Requests must be made within 24 hours.
## Missing items
Report missing items in the app within 48 hours. We refund the item price to your original payment method.
"""
for c in chunk_document(doc, "policy-17", "Help centre", max_tokens=25, overlap=5):
    print(c["id"], "|", c["text"].replace("\n", " / "))

def assemble_context(passages, budget_tokens, max_per_doc=2):
    """passages: dicts with id, doc_id, text, score, sorted best first."""
    chosen, used, per_doc, seen = [], 0, {}, set()
    for p in passages:
        key = p["text"].strip().lower()
        if key in seen or per_doc.get(p["doc_id"], 0) >= max_per_doc:
            continue
        cost = n_tokens(p["text"])
        if used + cost > budget_tokens:
            continue
        chosen.append(p); used += cost; seen.add(key)
        per_doc[p["doc_id"]] = per_doc.get(p["doc_id"], 0) + 1
    return "\n\n".join(f'<doc id="{p["id"]}">\n{p["text"]}\n</doc>' for p in chosen), [p["id"] for p in chosen]

ps = [{"id": "a#0", "doc_id": "a", "text": "voucher 20% late", "score": .9},
      {"id": "a#1", "doc_id": "a", "text": "voucher 20% late", "score": .8},
      {"id": "b#0", "doc_id": "b", "text": "missing items refund in 48 hours", "score": .7}]
ctx, ids = assemble_context(ps, budget_tokens=50)
print(ids); print(ctx)

def check_citations(answer, allowed_ids, passages_by_id, min_overlap=0.5):
    """Every [id] cited must exist in the context, and the sentence citing it should overlap its text."""
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        cited = re.findall(r"\[([^\]]+)\]", sentence)
        if not cited:
            problems.append(("uncited", sentence))
            continue
        for cid in cited:
            if cid not in allowed_ids:
                problems.append(("unknown id", cid))
                continue
            words = set(re.findall(r"\w+", re.sub(r"\[[^\]]+\]", "", sentence).lower()))
            src = set(re.findall(r"\w+", passages_by_id[cid].lower()))
            if words and len(words & src) / len(words) < min_overlap:
                problems.append(("weak support", sentence))
    return problems

pb = {"a#0": "Vouchers are worth 20% of the order value, up to 10 euros.",
      "b#0": "Report missing items in the app within 48 hours."}
ans = "You get a voucher worth 20% of the order value [a#0]. Missing items must be reported within 48 hours [b#0]. Refunds take 3 days [c#9]."
print(check_citations(ans, {"a#0", "b#0"}, pb))
