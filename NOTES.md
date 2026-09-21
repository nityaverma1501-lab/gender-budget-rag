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

I first tried Qwen2.5-0.5B-Instruct (chosen because this sandbox's network
made downloading anything larger unreliable). On a test question ("Self
Defence for Girls Students in Schools," Delhi 2024-25, correct answer Rs
20,000 thousand) it returned the bare string `"76000"` — wrong, and not
even in the format I'd asked for. Later, on Apple's on-device model, I hit
a subtler version: for "grand total of Parts A and B in BE 2012-13," it
read the *wrong* number out of a nine-column XLS row (97,134 instead of
88,143, the adjacent year's total) despite a clear prose sentence in the
same context stating the correct figure. Comparing against
`questions_dev.jsonl`'s gold answers caught both; neither was visible from
output validity or model confidence alone.

## What I'd do next

- Explicit column-position parsing for the borderless Delhi tables instead
  of relying on the LLM to infer columns from page text.
- A real eval harness scoring retrieval hit-rate and answer accuracy
  automatically, rather than spot-checking by hand.
- Extend the jurisdiction guardrail beyond its current hand-built
  Delhi/Odisha/Bihar list to detect any named Indian state.
- This sandbox's HF downloads were badly throttled and its Foundation Model
  subprocess occasionally hung for minutes — environment quirks I worked
  around; a normal machine or a hosted API key would likely be faster and
  more consistent.
