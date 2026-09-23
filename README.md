# Gender Budget RAG

A retrieval-augmented question-answering system over eleven Indian government
gender budget documents (PDFs, a CSV, and an XLS — see `data/`).

See [`SCOPE.md`](SCOPE.md) for what's in/out of scope and known corpus quirks,
and [`NOTES.md`](NOTES.md) for how the system works and what I'd do next.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Generation backend** — no API key needed either way:

- **macOS 26+ with Apple Intelligence enabled (primary path):** build the
  on-device generation bridge once:
  ```bash
  cd src
  swiftc -O fm_answer.swift -o fm_answer
  ```
  `generate.py` detects the compiled `fm_answer` binary and uses Apple's
  on-device Foundation Model — no download, runs locally, fast.
- **Any other machine (fallback path):** skip the `swiftc` step. `generate.py`
  falls back to a small local Hugging Face model
  (`Qwen/Qwen2.5-0.5B-Instruct`, ~1GB) run via `transformers`, downloaded
  automatically on first use. On the dev set both backends land in a similar,
  modest accuracy range (see `NOTES.md` for the exact breakdown) with
  different failure patterns rather than one being clearly better — neither
  is highly reliable on the messier tables in this corpus.

## Running the pipeline

From a clean clone, in order:

```bash
cd src

# 1. Extract chunks (text + metadata) from every file in ../data into
#    ../index/chunks.jsonl
python3 ingest.py

# 2. Embed all chunks with a multilingual sentence-transformer and save
#    ../index/embeddings.npy
python3 index.py

# 3. Answer a batch of questions (JSONL in, JSONL out in the assignment's
#    answers.jsonl schema)
python3 query.py ../questions_dev.jsonl ../answers_dev.jsonl --debug
python3 query.py ../questions_eval.jsonl ../answers.jsonl
```

`query.py --debug` also writes a `.debug.jsonl` file alongside the output
with the raw model output, retrieval scores, and which guardrail (if any)
triggered an abstention — useful for diagnosing wrong or abstained answers.

### Trying it interactively

```bash
cd src
python3 ask.py "What was Delhi's outlay for Ladli Yojna in 2024-25?"

# or, with no argument, drop into a REPL:
python3 ask.py
```

## How it works

1. **Ingest** (`src/ingest.py`) — file-type-specific extractors turn the CSV,
   XLS, and seven PDFs into a flat list of text chunks with `file`/`page`/`row`
   metadata and a per-document unit hint (crore / lakh / thousand).
2. **Index** (`src/index.py`) — chunks are embedded with
   `intfloat/multilingual-e5-base`, chosen specifically because it embeds
   Hindi and English into a shared space, so a Devanagari-script question can
   retrieve the right English-language table row even when the source PDF's
   Hindi text is unusable (see `SCOPE.md`).
3. **Retrieve** (`src/retrieve.py`) — hybrid search: dense cosine similarity
   plus BM25 keyword scoring, combined with a weighted sum, capped so a
   single long document can't crowd every result out of the top-k.
4. **Generate** (`src/generate.py`) — a local LLM reads the top retrieved
   excerpt and produces a two-line answer + citation, or abstains when the
   excerpt doesn't support an answer.
5. **Gate** (`src/query.py`) — three checks run before/around generation:
   a minimum retrieval-similarity floor, a deterministic check that a
   year named in the question actually appears in the retrieved text, and
   a deterministic check that a named jurisdiction (Delhi/Odisha/Bihar)
   matches the source document — each abstains without asking the LLM when
   it fails, rather than relying solely on the model to notice a mismatch.
