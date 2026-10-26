"""Export a Markdown report to .md and .pdf (Task 4)."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "report"


def strip_code_fence(md: str) -> str:
    """LLMs sometimes wrap the whole report in ```markdown ... ``` - remove that."""
    m = re.match(r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", md, flags=re.S)
    return m.group(1) if m else md


def markdown_to_pdf(md_text: str, pdf_path: str | Path) -> Path:
    import markdown as md_lib
    from fpdf import FPDF

    # Core PDF fonts are Latin-1 only: map common Unicode punctuation, drop the rest.
    table = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
             "—": "-", "…": "...", "•": "-", " ": " ", "→": "->"}
    safe = "".join(table.get(ch, ch) for ch in md_text).encode("latin-1", "ignore").decode("latin-1")
    html = md_lib.markdown(safe, extensions=["sane_lists"])
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.write_html(html)
    pdf.output(str(pdf_path))
    return Path(pdf_path)


def save_report(md_text: str, topic: str, out_dir: str | Path = "reports") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M}-{slugify(topic)}"
    md_text = strip_code_fence(md_text).strip() + "\n"
    md_path = out / f"{stem}.md"
    md_path.write_text(md_text, encoding="utf-8")
    pdf_path = markdown_to_pdf(md_text, out / f"{stem}.pdf")
    return {"markdown": md_path, "pdf": pdf_path}
