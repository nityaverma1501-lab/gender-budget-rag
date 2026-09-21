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
`transformers` on non-Apple-Intelligence machines.

## An AI mistake I caught

The scariest mistake wasn't a garbled number or a JSON parsing failure — it
was a fully confident, correctly-formatted answer that was simply wrong.

I queried the system for Delhi's Ladli Yojna allocation in 2019-20, a year
absent from my corpus entirely (the Delhi documents jump from 2011-12
straight to 2022-23). A well-behaved RAG system should abstain here.
Instead, the retriever pulled a real document about the same scheme from a
*different* year — the embedding similarity between "Ladli Yojna 2019-20"
and "Ladli Yojna 2022-23" was high enough to clear my confidence threshold
— and the LLM answered as if it were fact: specific number, specific
citation, zero hedging. This is a textbook RAG failure mode: retrieval
optimizes for semantic similarity, not correctness, so a
topically-close-but-wrong-year document scores just as high as a genuinely
correct one, and the model has no built-in sense of "the topic matches but
a key entity doesn't."

That silent-but-confident failure worried me more than any outright
hallucination, since an answer that *sounds* uncertain gets double-checked
while one that sounds authoritative gets trusted. I fixed it by moving the
check out of the model into deterministic pre-generation logic: the
pipeline now extracts the year (and jurisdiction) named in the question and
verifies that exact value appears in the retrieved chunk's text before the
LLM ever runs. If it doesn't match, the system abstains outright — no model
call needed. That one guardrail converted several silent wrong answers into
honest `"answered": false` responses.

## What I'd do next

- Explicit column-position parsing for the borderless Delhi tables instead
  of relying on the LLM to infer columns from page text.
- A real eval harness scoring retrieval hit-rate and answer accuracy
  automatically, rather than spot-checking by hand.
- Extend the jurisdiction guardrail beyond its current hand-built
  Delhi/Odisha/Bihar list to detect any named Indian state.
