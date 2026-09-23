"""
Ingestion: turn the 11 raw corpus files into a flat JSONL of retrieval chunks.

Each chunk is a dict:
  {
    "id": unique string,
    "file": source filename,
    "page": 1-indexed PDF page number, or null for spreadsheet chunks,
    "row": spreadsheet row index, or null for PDF chunks,
    "text": the chunk text,
    "unit_hint": a short note on the document's reporting unit, to keep the
                 generation model from mixing up crore / lakh / thousand,
  }

Known corpus quirks handled here (see SCOPE.md for the full list):
  - MRF_13_Union_Budget.csv is latin-1 encoded, not UTF-8.
  - stat20.xls has its real header several rows down, a narrative note in
    free-text cells above the table, and a "GRAND TOTAL" row at the bottom.
  - The Delhi PDFs render Hindi in a legacy (non-Unicode) font, so the
    extracted Hindi text is unsearchable glyph garbage. We keep it (it's
    harmless) but rely on the English lines for retrieval and answers.
  - gender_budget_2011-12.pdf and gender_budget_2012-13.pdf are byte-identical
    duplicates; the content is actually the 2011-12 statement (RE 2010-11 /
    Annual Plan 2011-12). We tag both with their true content year so the
    query layer can recognize that no genuine 2012-13 Delhi document exists.
"""
import json
import re
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "index" / "chunks.jsonl"

# gender_budget_2012-13.pdf is a mislabeled duplicate of gender_budget_2011-12.pdf.
CONTENT_YEAR_OVERRIDE = {
    "gender_budget_2012-13.pdf": (
        "NOTE: despite its filename, this file's actual content is the "
        "2011-12 statement (Revised Estimates 2010-11 / Annual Plan 2011-12) "
        "-- it is a byte-identical duplicate of gender_budget_2011-12.pdf. "
        "There is no genuine year-13-labeled Delhi gender budget document in this corpus."
    )
}


def make_chunk(file, text, page=None, row=None, unit_hint=None, extra_note=None):
    text = text.strip()
    if extra_note:
        text = f"[{extra_note}]\n{text}"
    return {
        "file": file,
        "page": page,
        "row": row,
        "text": text,
        "unit_hint": unit_hint,
    }


def extract_pdf(path, unit_hint=None):
    doc = pymupdf.open(path)
    chunks = []
    note = CONTENT_YEAR_OVERRIDE.get(path.name)
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        chunks.append(make_chunk(path.name, text, page=i + 1, unit_hint=unit_hint, extra_note=note))
    return chunks


def extract_union_csv(path):
    df = pd.read_csv(path, encoding="latin-1")
    chunks = []
    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        text = "Union Gender Budget Statement (MRF-13) row -- " + " | ".join(parts)
        chunks.append(make_chunk(path.name, text, row=int(idx), unit_hint="crore"))
    return chunks


_NUM_RE = re.compile(r"^-?[\d,]+(\.\d+)?$")


def _is_number(s):
    return bool(_NUM_RE.match(s.replace(",", "").strip())) if s.strip() else False


def extract_stat20_xls(path):
    raw = pd.read_excel(path, sheet_name=0, header=None)
    chunks = []

    # 1) Narrative free-text cells (long sentences) anywhere in the sheet --
    #    this is where facts like "30 Ministries/Departments and 5 UTs" and
    #    the grand-total prose live, outside the tabular region.
    narrative_cells = []
    for i in range(len(raw)):
        for v in raw.iloc[i]:
            if isinstance(v, str) and len(v.strip()) > 60:
                narrative_cells.append(v.strip())
    if narrative_cells:
        text = "Statement 20 (Gender Budget), Sheet1 narrative notes:\n" + "\n".join(narrative_cells)
        chunks.append(make_chunk(path.name, text, unit_hint="crore (unless stated otherwise)"))

    # 2) Tabular region: forward-fill the nearest non-numeric label rows as
    #    context (ministry / demand headings are on their own rows, split
    #    across merged cells) and treat any row with >=2 numeric cells as a
    #    data row. The column headers (which BE/RE year each of the 9 number
    #    columns is) sit many rows above the data and would otherwise fall
    #    out of the sliding context window long before we reach, e.g., the
    #    grand-total row near the bottom -- without them a row like
    #    "88142.8 | 73314.76 | ..." is just an ambiguous list of numbers, so
    #    we carry the year/Plan-Non-Plan-Total header forward into every row.
    column_header = None
    context = []
    for i in range(len(raw)):
        cells = [str(x).strip() for x in raw.iloc[i].tolist() if str(x).strip() not in ("nan", "")]
        if not cells:
            continue
        numeric_count = sum(1 for c in cells if _is_number(c))
        if numeric_count >= 2:
            label = " / ".join(context[-4:])
            header_note = f"Columns in order: {column_header} | " if column_header else ""
            text = (
                f"Statement 20 table row -- Context: {label} | {header_note}"
                f"Values: {' | '.join(cells)}"
            )
            chunks.append(make_chunk(path.name, text, row=i, unit_hint="crore"))
        else:
            joined = " ".join(cells)
            if "MINISTRY/DEPARTMENT" in joined.upper():
                # this row + the year row directly above it form the column header
                year_row_idx = i - 1
                year_cells = [
                    str(x).strip()
                    for x in raw.iloc[year_row_idx].tolist()
                    if str(x).strip() not in ("nan", "")
                ]
                column_header = " / ".join(year_cells) + " -- " + joined
            context.append(joined)
            if len(context) > 8:
                context.pop(0)
    return chunks


def main():
    all_chunks = []

    all_chunks += extract_union_csv(DATA_DIR / "MRF_13_Union_Budget.csv")
    all_chunks += extract_stat20_xls(DATA_DIR / "stat20.xls")

    delhi_files = [
        "gender_budget_2011-12.pdf",
        "gender_budget_2012-13.pdf",
        "gender_budget_2022-23.pdf",
        "gender_budget_2023-24.pdf",
        "gender_budget_2024-25.pdf",
        "gender_budget_2025-26.pdf",
        "gender_budget_2026-27.pdf",
    ]
    for f in delhi_files:
        all_chunks += extract_pdf(DATA_DIR / f, unit_hint="Rs in thousands (Delhi gender budget)")

    all_chunks += extract_pdf(DATA_DIR / "14-Gender_Budget.pdf", unit_hint="Rs in lakh (Odisha gender budget)")
    all_chunks += extract_pdf(
        DATA_DIR / "17107782611749468962.pdf",
        unit_hint="bilingual (English/Hindi) Bihar gender budget; figures as stated in the English text",
    )

    for i, c in enumerate(all_chunks):
        c["id"] = f"{i:05d}"

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {OUT_PATH}")
    by_file = {}
    for c in all_chunks:
        by_file[c["file"]] = by_file.get(c["file"], 0) + 1
    for f, n in by_file.items():
        print(f"  {f}: {n} chunks")


if __name__ == "__main__":
    main()
