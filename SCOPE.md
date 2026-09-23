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
- **Guardrails check years and jurisdictions, not scheme names.** A
  question about a scheme that isn't in the corpus can still get an answer:
  E15 asks about a "Mahila Samriddhi Yojana" that appears nowhere in the
  Odisha document, and gets an unrelated Odisha figure. Uncovered states
  (e.g. Kerala, E13) currently abstain, but that relies on the model
  declining rather than on a deterministic check.
- **Dev-set accuracy is 8/12 fully correct (67%)** — the remaining wrong
  answers (D05, D06, D11) are the LLM misreading a number off a dense
  multi-column table even when given the right page. D10 is 2 of 3
  figures. See `NOTES.md` for the exact breakdown.
- **The fallback backend is much weaker.** Those 8/12 are on the Apple
  on-device model. On a machine without Apple Intelligence the code falls
  back to Qwen2.5-0.5B, which scores 2/12 strictly (up to 6/12 counting
  right-number-missing-unit answers), so a reviewer running it elsewhere
  should expect different, worse answers than the committed
  `answers.jsonl`.
- No OCR; assumes embedded text layers (true for all 11 files here).
