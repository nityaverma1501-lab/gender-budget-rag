# Notes

## How it works

`ingest.py` turns the 11 files into ~940 metadata-tagged text chunks
(file/page/row + a unit hint). `index.py` embeds them with a multilingual
sentence-transformer. `retrieve.py` does hybrid dense+BM25 search with a
per-file cap so one long document can't dominate the results. `query.py`
adds two deterministic guardrails on top of retrieval (year match,
jurisdiction match — see `SCOPE.md`) before ever calling the LLM.
`generate.py` runs a local LLM on the single best-matching excerpt and
returns a short answer + citation, or an explicit abstention.

Generation defaults to Apple's on-device Foundation Model (Apple
Intelligence), called through a small compiled Swift bridge
(`src/fm_answer.swift`) — no API key, no download, genuinely private. It has
a 4096-token context window shared by the system prompt, the excerpt, and
the reply, which forced a much tighter, single-excerpt prompt than I'd
originally planned; `generate.py` falls back to a downloaded open model via
`transformers` (Qwen2.5-0.5B-Instruct) on non-Apple-Intelligence machines.

## Honest accuracy

On the 12-question dev set, comparing against the gold answers: **4/12 are
fully correct** (D01, D03, D08, D09), **1/12 partially correct** (D02 —
returns only one of two legitimate figures), **5/12 wrong**, and **2/12
falsely abstain** when the corpus does contain the answer. I checked this
by running the fallback Qwen backend on the same 12 questions too, since a
reviewer without Apple Intelligence will get that path, not the one I
mostly developed against: it lands in a similar overall range (4/12 fully
correct, but a different 4 — D04 instead of D08 — plus 3 partial and 4
wrong), so the two backends aren't reliably interchangeable question-by-
question even though their aggregate accuracy is close. Most of the wrong
answers are numeric misreads on dense tables (see below), not retrieval
failures — the right document is usually being found.

## An AI mistake I caught

The scariest mistake wasn't a garbled number — it was a fully confident,
correctly-formatted answer that was simply wrong.

I queried the system for Delhi's Ladli Yojna allocation in 2019-20, a year
absent from my corpus entirely (the Delhi documents jump from 2011-12
straight to 2022-23). Instead of abstaining, the retriever pulled a real
document about the same scheme from a *different* year — embedding
similarity between "Ladli Yojna 2019-20" and "...2022-23" cleared my
confidence threshold — and the LLM answered as fact: specific number,
specific citation, zero hedging. Textbook RAG failure: retrieval optimizes
for semantic similarity, not correctness, so a topically-close-but-wrong-
year document scores as high as a genuinely correct one, and the model has
no built-in sense that "the topic matches but a key entity doesn't."

That worried me more than any outright hallucination, since an answer that
*sounds* uncertain gets double-checked while one that sounds authoritative
gets trusted. I fixed it with deterministic pre-generation logic: the
pipeline now extracts the year (and jurisdiction) named in the question and
verifies that value appears in the retrieved text before the LLM ever runs.
If it doesn't, the system abstains outright — no model call needed. That
one guardrail converted several silent wrong answers into honest
`"answered": false` responses.

## What I'd do next

- Explicit column-position parsing for the borderless Delhi tables and the
  9-column XLS rows, instead of relying on the LLM to infer columns from
  page text — this is the single biggest source of wrong numbers.
- A real eval harness scoring retrieval hit-rate and answer accuracy
  automatically against the dev set, rather than spot-checking by hand.
- Extend the jurisdiction guardrail (currently a hand-built
  Delhi/Odisha/Bihar/Union list) to detect any named Indian state, so an
  uncovered one (e.g. Kerala) is caught by name instead of relying on the
  LLM alone, which doesn't reliably catch it.
- Investigate why identical runs of the same question occasionally return
  different answers (observed on D07) — likely batching/scheduling
  non-determinism in the on-device model rather than my code, but I haven't
  isolated it.
