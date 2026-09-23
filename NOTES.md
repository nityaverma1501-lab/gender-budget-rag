# Notes

## How it works

`ingest.py` turns the 11 files into ~940 metadata-tagged text chunks
(file/page/row + a unit hint). `index.py` embeds them with a multilingual
sentence-transformer, falling back to dense-only scoring for Devanagari
queries (BM25's tokenizer can't score Hindi text, only its stray digits).
`query.py` runs three deterministic guardrails before ever calling the LLM
— a similarity floor, a year match, and a jurisdiction match
(Delhi/Odisha/Bihar/Union, see `SCOPE.md`) — and deterministically
assembles the answer, bypassing the LLM, for one recurring structured case
(a scheme listed under both Part A and Part B of the Union CSV) where the
model kept silently dropping one figure. `generate.py` runs a local LLM on
the best-matching excerpt(s) and returns a short answer + citation or an
explicit abstention; it also discards any citation the model fabricates to
a file it was never actually shown.

Generation defaults to Apple's on-device Foundation Model, called through a
compiled Swift bridge (`src/fm_answer.swift`) — no API key, no download.
Its 4096-token context window forced a tight, mostly single-excerpt prompt;
`generate.py` falls back to a local Qwen2.5-0.5B model on machines without
Apple Intelligence.

## Honest accuracy

On the 12-question dev set: **7/12 fully correct** (D01, D02, D03, D04,
D08, D09, D12), **1/12 substantially correct** (D10 — 2 of 3 required
figures exactly right), **4/12 wrong** (D05, D06, D07, D11 — numeric
misreads on dense multi-column tables, not retrieval misses). Zero false
abstentions.

Getting here from an earlier 4/12 took real debugging: a missing "Union"
jurisdiction let Union-budget questions retrieve from an unrelated state's
document; Bihar/Odisha (each a single-year document) were wrongly held to
a year-check meant for Delhi's multi-year PDFs; and BM25 was quietly
sabotaging Hindi queries by scoring on stray digits alone.

I re-ran the same 12 questions through the fallback Qwen backend too
(what a reviewer without Apple Intelligence gets), with the same fixes
applied. It lands at roughly 3-5/12 — clearly behind. With retrieval now
reliably surfacing the right excerpt, the remaining gap is squarely the
generation model's numeric-reading ability, which 0.5B parameters isn't
enough for on this corpus's dense tables.

## An AI mistake I caught

The scariest mistake wasn't a garbled number — it was a fully confident,
correctly-formatted answer that was simply wrong.

I queried the system for Delhi's Ladli Yojna allocation in 2019-20, a year
absent from my corpus (the Delhi documents jump from 2011-12 straight to
2022-23). Instead of abstaining, the retriever pulled a real document about
the same scheme from a *different* year — embedding similarity between
"Ladli Yojna 2019-20" and "...2022-23" cleared my confidence threshold —
and the LLM answered as fact: specific number, specific citation, zero
hedging. Textbook RAG failure: retrieval optimizes for similarity, not
correctness, and the model has no built-in sense that "the topic matches
but a key entity doesn't."

That worried me more than any outright hallucination, since an answer that
*sounds* uncertain gets double-checked while one that sounds authoritative
gets trusted. I fixed it with deterministic pre-generation logic: the
pipeline extracts the year (and jurisdiction) named in the question and
verifies that value appears in the retrieved text before the LLM runs. If
it doesn't, the system abstains — no model call needed.

## What I'd do next

- Explicit column-position parsing for the borderless Delhi tables and
  9-column XLS rows, instead of relying on the LLM to infer columns from
  raw page text — this is the single biggest source of remaining errors.
- A real eval harness scoring accuracy automatically against the dev set.
- Extend the jurisdiction guardrail beyond its hand-built
  Delhi/Odisha/Bihar/Union list to detect any named Indian state, so an
  uncovered one (e.g. Kerala, currently answered from Odisha's data
  instead of abstaining) is caught by name.
