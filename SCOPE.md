# Scope

## Included

All 11 corpus files are ingested: the CSV (row-level chunks, one per scheme
plus Part/Grand-Total rows), the XLS (narrative free-text cells extracted
separately from the tabular region, whose real header sits 11 rows down with
a `GRAND TOTAL` row at the very bottom — every data-row chunk carries the
column-year header forward so a bare list of nine numbers stays
interpretable), and the seven Delhi PDFs plus the Odisha and Bihar PDFs
(page-level chunks, page numbers preserved for citation). Retrieval is
hybrid: dense multilingual embeddings (`intfloat/multilingual-e5-base`) plus
BM25, with a per-source-file cap so one long prose document can't crowd a
short-but-correct spreadsheet row out of the results. Two deterministic
guardrails sit in front of generation: if the question names a specific
financial year or one of the four covered jurisdictions (Delhi/Odisha/
Bihar/Union), the system requires the answering excerpt to actually match
that year/jurisdiction, abstaining otherwise rather than trusting the LLM
alone to notice a mismatch — added after the model was observed confidently
answering a "2019-20" question from a real-but-wrong-year 2022-23 document,
and separately answering Union-budget questions from an Odisha document
because nothing checked that "Union" and "Odisha" aren't the same thing.

## Excluded / handled imperfectly

- **Delhi PDFs are borderless tables.** No explicit column reconstruction
  from word coordinates; each page is one chunk (header + all scheme rows in
  reading order) and the LLM infers which number belongs to which column
  from the header text on the same page. On dense pages with many similar
  numbers this sometimes misreads a figure (an observed, documented failure
  mode — see `NOTES.md`).
- **Hindi text in the Delhi/Bihar PDFs uses a legacy non-Unicode font** and
  extracts as unsearchable glyph garbage; the multilingual embedding model
  matches a Devanagari-script query onto the correct English-language row
  instead, since that's where the real figures are.
- **`gender_budget_2011-12.pdf` / `gender_budget_2012-13.pdf` are
  byte-identical**; true content is 2011-12. Flagged in chunk metadata.
- **XLS multi-row merged labels** are reconstructed by forward-filling
  nearby non-numeric rows — heuristic, not guaranteed for every row.
- **Jurisdiction guardrail only covers Delhi/Odisha/Bihar/Union** (the
  jurisdictions this corpus actually has). A question naming an uncovered
  state (e.g. Kerala) isn't caught by that specific check and relies on the
  LLM alone to recognize the corpus doesn't cover it, which it doesn't
  always do — E13 in `answers.jsonl` asks about Kerala and gets Odisha's
  number back instead of an abstention, a known, documented miss.
- **Dev-set accuracy is 7/12 fully correct, not higher** — the remaining
  4 wrong answers (D05, D06, D07, D11) are the LLM misreading a number out
  of a dense multi-column table, not retrieval finding the wrong document.
  See `NOTES.md` for the exact breakdown.
- **The fallback backend is weaker.** Those 7/12 are on the Apple
  on-device model. On a machine without Apple Intelligence the code falls
  back to Qwen2.5-0.5B, which scores roughly 3-5/12 on the same questions,
  so a reviewer running it elsewhere should expect different (worse)
  answers than the committed `answers.jsonl`.
- No OCR; assumes embedded text layers (true for all 11 files here).
