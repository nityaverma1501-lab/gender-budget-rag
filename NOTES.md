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

On the 12-question dev set: **8/12 fully correct (67%)** — D01, D02, D03,
D04, D07, D08, D09, D12 — plus **D10 substantially correct** (2 of 3
required figures exactly right) and **3/12 wrong** (D05, D06, D11 — the
model misreading a number off a dense multi-column table even when given
the right page). Zero false abstentions. Two identical runs gave
byte-identical results.

Getting here from an earlier 4/12 took real debugging, not prompt tweaks:
a missing "Union" jurisdiction let Union-budget questions retrieve from an
unrelated state's document; Bihar/Odisha (each a single-year document)
were wrongly held to a year-check meant for Delhi's multi-year PDFs; BM25
was sabotaging Hindi queries by scoring on stray digits alone; whole-page
embeddings buried a single answer-bearing sentence ("Australia was the
first country...") under a neighbouring page, fixed by re-ranking
candidates on their best 3-line passage; and a page merely *mentioning*
"2024-25" (the 2025-26 PDF's Revised Estimates column) was outranking the
2024-25 document itself. Not every change helped: passage re-ranking
broke two Hindi questions that whole-page matching had right, so it's
switched off for Devanagari queries.

The fallback Qwen backend (what a reviewer without Apple Intelligence
gets) scores **2/12 strictly**, up to 6/12 if right-number-missing-unit
answers count. With retrieval now surfacing the right excerpt, the gap is
squarely the 0.5B model's numeric reading.

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
- Extend the year/jurisdiction guardrails to named schemes: if the scheme
  a question names appears nowhere in the retrieved text, abstain. E15
  asks about a "Mahila Samriddhi Yojana" that isn't in the Odisha document
  at all, and the system still answers with an unrelated figure.
- Detect any named Indian state, not just the hand-built
  Delhi/Odisha/Bihar/Union list. Kerala (E13) currently abstains, but
  because the model declined, not because a check caught it.
