"""Local answer generation, no API key required.

Backend 1 (preferred): Apple's on-device Foundation Model (Apple Intelligence),
via a small compiled Swift CLI (fm_answer). Runs fully on-device, needs no
download, and is available on macOS 26+ with Apple Intelligence enabled.

Backend 2 (fallback): a local Hugging Face model (Qwen2.5-Instruct) run via
transformers, for machines without Apple Intelligence. Picks the largest
fully-downloaded model in .model_cache/, else downloads the smallest (0.5B).
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FM_BINARY = Path(__file__).resolve().parent / "fm_answer"

SYSTEM_PROMPT = """Answer questions about Indian gender budget documents using ONLY the given excerpts. \
Never use outside knowledge. If the excerpts don't clearly contain the answer, reply exactly: \
ANSWER: NOT_FOUND
Give a COMPLETE answer: include the exact number AND its unit (crore/lakh/thousand) AND any \
qualifying detail the question asks for. If excerpts show similar data for different years or \
documents, use ONLY the one matching exactly what the question asks (check the year/jurisdiction \
carefully) and ignore the rest.
Reply in EXACTLY two lines, no other text:
ANSWER: <a short direct answer, or NOT_FOUND>
SOURCES: <comma-separated "file p.N">

Example:
ANSWER: Rs 20,000 thousand (Rs 2 crore).
SOURCES: gender_budget_2024-25.pdf p.1
"""

_SOURCE_RE = re.compile(r"([^,;]+?\.(?:pdf|csv|xls))\s*(?:p\.?\s*(\d+))?", re.IGNORECASE)


def _parse_reply(text, fallback_chunks):
    text = text.strip()
    answer_match = re.search(r"ANSWER:\s*(.*)", text)
    sources_match = re.search(r"SOURCES:\s*(.*)", text)

    if answer_match:
        raw_answer = answer_match.group(1).split("\n")[0].strip()
    else:
        # Small models sometimes skip the "ANSWER:" label and just emit the
        # answer text directly -- fall back to treating the whole reply as
        # the answer rather than discarding a correct extraction over a
        # missing label.
        raw_answer = text.split("\n")[0].strip()

    if not raw_answer or raw_answer.upper().startswith("NOT_FOUND"):
        return {"answered": False, "answer": "", "sources": []}

    # The model occasionally echoes the "(file=X, page=Y)" excerpt-header
    # format from the prompt back into its answer instead of using the
    # SOURCES: line -- strip that artifact rather than leaking it to the user.
    raw_answer = re.sub(r"[<(]file=.*?[)>]", "", raw_answer).strip()
    # It has also been observed echoing the literal "<placeholder>" text
    # from the prompt's own instructions/example into its answer -- strip
    # any bracketed placeholder-looking fragment for the same reason.
    raw_answer = re.sub(r"<[^<>]*(?:answer|NOT_FOUND|file|page)[^<>]*>", "", raw_answer, flags=re.IGNORECASE).strip()
    # A bare unit with no number, e.g. "2005-06 (crore)" or "10% (Rs. crore)"
    # -- the model tacking a unit onto a year or percentage. Drop it.
    raw_answer = re.sub(
        r"\s*\((?:Rs\.?\s*)?(?:crore|lakh|thousand)s?\)", "", raw_answer, flags=re.IGNORECASE
    ).strip()

    known_files = {c["file"] for c in fallback_chunks} if fallback_chunks else set()

    sources = []
    if sources_match:
        sources_line = sources_match.group(1).split("\n")[0].strip()
        for m in _SOURCE_RE.finditer(sources_line):
            fname = m.group(1).strip()
            page = int(m.group(2)) if m.group(2) else None
            # The model has been observed to cite a filename it was never
            # actually shown (e.g. one from an unrelated jurisdiction) --
            # a fabricated citation is worse than none, so only keep a
            # cited source if it matches a chunk that was really in its
            # context.
            if not known_files or fname in known_files:
                sources.append({"file": fname, "page": page})

    if not sources and fallback_chunks:
        # The model gave a real answer but didn't cite cleanly (or cited
        # something fabricated, filtered out above) -- fall back to the
        # single highest-ranked retrieved chunk as the source.
        top = fallback_chunks[0]
        sources.append({"file": top["file"], "page": top.get("page")})

    return {"answered": True, "answer": _fix_thousand_to_crore(raw_answer), "sources": sources}


_THOUSAND_RE = re.compile(
    r"(\d[\d,]*)\s*thousand\s*(?:\((?:Rs\.?\s*)?[\d.,]+\s*(?:crore|lakh)\)?)?", re.IGNORECASE
)


def _fix_thousand_to_crore(text):
    """LLMs are unreliable at unit-conversion arithmetic (observed: model
    converted "20,000 thousand" to "Rs 20 lakh" instead of the correct
    Rs 2 crore). 1 crore = 1,00,00,000 = 10,000 thousand, so recompute this
    deterministically in Python rather than trusting the model's math, and
    replace any (possibly wrong) conversion the model already wrote."""

    def repl(m):
        raw_num = m.group(1).replace(",", "")
        try:
            thousands = float(raw_num)
        except ValueError:
            return m.group(0)
        crore = thousands / 10000
        crore_str = f"{crore:.2f}".rstrip("0").rstrip(".") if crore != int(crore) else str(int(crore))
        return f"{m.group(1)} thousand (Rs {crore_str} crore)"

    return _THOUSAND_RE.sub(repl, text)


def _truncate_on_boundary(text, limit):
    """Cut at the last newline before `limit` rather than mid-line, so a
    borderless-table row (scheme name on one line, its numbers on the next)
    doesn't get split in half -- a raw text[:limit] cut was observed to
    occasionally sever a row from the numbers that belong to it."""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    return text[: cut if cut > limit * 0.5 else limit]


_ROW_CHUNK_LEN = 600  # below this, treat a chunk as a short CSV/XLS row, not a PDF page


def build_context(chunks, total_char_budget=3200, max_chunks=None):
    # Apple's on-device model has a 4096-token context window, shared with
    # the system instructions and the question. Rather than a flat per-chunk
    # cap (which either wastes budget on short row-based chunks or cuts a
    # correct-but-lower-ranked chunk out of context entirely), spend a
    # shared character budget greedily in rank order: short chunks (CSV/XLS
    # rows) are included in full essentially for free, leaving more room for
    # further-down-the-list chunks to fit too; a large page chunk that would
    # blow the remaining budget is truncated to what's left, and we stop
    # once the budget runs out. Chunks are expected to already be filtered/
    # reordered by the caller's guardrails (year/jurisdiction match), so
    # chunks[0] is the best-evidenced excerpt, not just the highest raw
    # similarity score.
    #
    # max_chunks defaults adaptively: a dense borderless PDF page confuses
    # the model when a second, similarly-dense page is added alongside it
    # (observed: correct single-page answers turned wrong once a second
    # page's numbers entered the same prompt) -- so those get capped to 1.
    # But a short CSV/XLS row is a clean "Category: X | Scheme: Y | value: Z"
    # line with no such risk, and some questions legitimately need two of
    # them (e.g. a scheme listed under both Part A and Part B) -- those get
    # room for several.
    if max_chunks is None:
        max_chunks = 4 if chunks and len(chunks[0]["text"]) < _ROW_CHUNK_LEN else 1

    parts = []
    remaining = total_char_budget
    for c in chunks[:max_chunks]:
        if remaining <= 200:  # not enough left for a useful excerpt
            break
        loc = f"file={c['file']}"
        if c.get("page") is not None:
            loc += f", page={c['page']}"
        if c.get("unit_hint"):
            loc += f", unit_hint={c['unit_hint']}"
        text = _truncate_on_boundary(c["text"], remaining)
        remaining -= len(text)
        parts.append(f"--- Excerpt ({loc}) ---\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Backend 1: Apple on-device Foundation Model
# ---------------------------------------------------------------------------

def _fm_available():
    return FM_BINARY.exists()


def _fm_generate(instructions, prompt, timeout=25):
    payload = json.dumps({"instructions": instructions, "prompt": prompt})
    try:
        proc = subprocess.run(
            [str(FM_BINARY)], input=payload, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        # Occasionally hangs talking to the on-device model daemon; fail
        # fast so the caller's retry/fallback logic can move on rather than
        # waiting the full subprocess timeout on every stalled attempt.
        raise RuntimeError(f"fm_answer timed out after {timeout}s")
    if proc.returncode != 0:
        raise RuntimeError(f"fm_answer failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# Backend 2: local transformers model (fallback)
# ---------------------------------------------------------------------------

_CANDIDATE_MODEL_DIRS = [
    ROOT / ".model_cache" / "Qwen2.5-3B-Instruct",
    ROOT / ".model_cache" / "Qwen2.5-1.5B-Instruct",
    ROOT / ".model_cache" / "Qwen2.5-0.5B-Instruct",
]


def _is_fully_written(path):
    """A safetensors shard is pre-truncated to its final size before the
    download fills it in, so a partial download still reports the full
    logical size via stat().st_size / os.path.getsize. Compare actual disk
    blocks used against the logical size instead, which only matches once
    every byte has actually been written."""
    if not path.exists():
        return False
    st = path.stat()
    actual_bytes = st.st_blocks * 512
    return actual_bytes >= st.st_size * 0.99


def _is_complete_model_dir(d):
    if not d.exists():
        return False
    if (d / ".complete").exists():
        return True
    if (d / "model.safetensors").exists():
        return _is_fully_written(d / "model.safetensors")
    index = d / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        shards = set(weight_map.values())
        return all(_is_fully_written(d / s) for s in shards)
    return False


HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
for _d in _CANDIDATE_MODEL_DIRS:
    if _is_complete_model_dir(_d):
        HF_MODEL_NAME = str(_d)
        break

_tokenizer = None
_model = None


def _hf_load():
    global _tokenizer, _model
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME, dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
    _model.eval()


def _hf_generate(instructions, prompt, max_new_tokens=300):
    import torch

    _hf_load()
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]
    chat_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(chat_prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tokenizer.eos_token_id,
        )
    return _tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


BACKEND = "apple_foundation_models" if _fm_available() else f"transformers:{HF_MODEL_NAME}"


def answer(question, chunks):
    used_fallback = False
    if _fm_available():
        # Retry on failure. Three distinct failure modes observed:
        # (1) context-window overflow -- shrinking the excerpt budget fixes
        #     it (the 4096-token budget covers instructions + excerpt +
        #     question + response together, and mixed English/Hindi text
        #     tokenizes less predictably than plain English);
        # (2) a transient hang talking to the on-device model daemon, which
        #     has nothing to do with content size -- retrying the *same*
        #     budget after a short pause usually succeeds;
        # (3) Apple's built-in content-safety guardrail refusing to answer
        #     at all -- observed on entirely benign government text (child
        #     malnutrition rates, menstrual hygiene programmes), which no
        #     amount of retrying or budget-shrinking fixes since it isn't a
        #     size or transient problem, it's a hard content-policy refusal.
        # For (3) especially, retrying forever would just keep failing, so
        # after exhausting retries we fall back to the local Qwen model
        # (which has no such guardrail) rather than silently abstaining on
        # a question the corpus actually answers.
        import time as _time

        last_error = None
        gen_text = None
        for attempt, budget in enumerate((3200, 3200, 1800, 900, 900)):
            if attempt > 0:
                _time.sleep(2)
            context = build_context(chunks, total_char_budget=budget)
            user_prompt = f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\nReply:"
            try:
                gen_text = _fm_generate(SYSTEM_PROMPT, user_prompt)
                break
            except Exception as e:
                last_error = e

        if gen_text is None:
            try:
                # Qwen has a much larger context window than Apple's 4096
                # tokens -- no need to starve it down to 1200 chars, which
                # was observed to cut the answer's actual numbers out of the
                # excerpt entirely and make the model invent plausible-
                # looking ones instead.
                context = build_context(chunks, total_char_budget=3200)
                user_prompt = f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\nReply:"
                gen_text = _hf_generate(SYSTEM_PROMPT, user_prompt)
                used_fallback = True
            except Exception:
                return {"answered": False, "answer": "", "sources": [], "_error": str(last_error)}
    else:
        context = build_context(chunks)
        user_prompt = f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\nReply:"
        gen_text = _hf_generate(SYSTEM_PROMPT, user_prompt)

    result = _parse_reply(gen_text, chunks)
    result["_raw"] = gen_text
    if used_fallback:
        result["_used_qwen_fallback"] = True
    return result
