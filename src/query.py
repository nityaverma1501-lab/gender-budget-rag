"""Top-level query pipeline: retrieve -> generate -> abstain-gate -> answers.jsonl record."""
import argparse
import json
import re
from pathlib import Path

from generate import answer as llm_answer
from retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent

# Matches a financial-year range like "2023-24", "2023-2024", or the "&"
# separator the legacy Hindi font in the Delhi PDFs uses ("2024&2025").
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})[-&](?:(?:19|20)?(\d{2}))\b")


def _normalize_years(text):
    """Extract financial-year ranges as a set of "STARTYEAR-XX" strings so
    "2023-24", "2023-2024" and "2023&2024" (legacy-font Delhi PDFs) all
    normalize to the same "2023-24" token and can be compared directly."""
    years = set()
    for start, end2 in _YEAR_RANGE_RE.findall(text):
        years.add(f"{start}-{end2}")
    return years


# This corpus covers exactly three state/UT jurisdictions (plus the Union
# CSV/XLS). If a question names one of these three (English or Hindi
# script), the answering chunk must come from that jurisdiction's own
# file(s) -- cross-lingual dense retrieval was observed to occasionally
# match a topically-similar page from a *different* state's document (e.g.
# a Hindi question about Delhi's Ladli Yojna retrieving Bihar-document pages
# that happen to share vocabulary), which the small local LLM then answered
# from without noticing the jurisdiction was wrong.
JURISDICTION_KEYWORDS = {
    "delhi": "delhi", "दिल्ली": "delhi",
    "odisha": "odisha", "orissa": "odisha", "ओड़िशा": "odisha", "उड़ीसा": "odisha",
    "bihar": "bihar", "बिहार": "bihar",
}
JURISDICTION_FILE_PREFIX = {
    "delhi": "gender_budget_",
    "odisha": "14-Gender_Budget.pdf",
    "bihar": "17107782611749468962.pdf",
}


def _question_jurisdiction(question):
    q_lower = question.lower()
    for kw, jurisdiction in JURISDICTION_KEYWORDS.items():
        if kw in q_lower or kw in question:
            return jurisdiction
    return None


def _chunk_matches_jurisdiction(chunk, jurisdiction):
    expected = JURISDICTION_FILE_PREFIX[jurisdiction]
    return chunk["file"].startswith(expected)

# If the best retrieved chunk's dense similarity is below this, nothing in
# the corpus is even topically related -- abstain without asking the LLM.
# This is deliberately a low bar (a *safety net*, not the main abstention
# mechanism): short factual questions about "Delhi", "budget", "allocation"
# etc. still score ~0.8+ against topically-similar-but-wrong chunks (e.g. a
# real Delhi document with the wrong year), so most "does the corpus not
# actually contain this?" judgement calls are left to the LLM, which sees
# the full excerpt text and is instructed to abstain when the specific fact
# asked for isn't in it.
MIN_DENSE_SCORE = 0.70
# Retrieve a fairly wide pool -- only 1 chunk actually reaches the model
# (generate.build_context caps at max_chunks=1), but the year/jurisdiction
# guardrails below need enough candidates to find a same-year,
# same-jurisdiction chunk even when it isn't in, say, the top 6 by raw
# similarity (observed: the correct Delhi 2022-23 page ranked ~12th for one
# Hindi-language query, behind several higher-scoring but wrong-year/
# wrong-jurisdiction pages).
TOP_K = 15


def run_query(retriever, question, top_k=TOP_K):
    chunks = retriever.search(question, top_k=top_k)
    best = chunks[0]["dense_score"] if chunks else 0.0

    if best < MIN_DENSE_SCORE:
        return {
            "answered": False,
            "answer": "",
            "sources": [],
            "_retrieval_gate": "below_threshold",
            "_best_score": best,
        }

    # Jurisdiction handling: if the question names one of the three
    # state/UT jurisdictions this corpus covers, require the chunks used
    # for generation to actually come from that jurisdiction's document(s).
    jurisdiction = _question_jurisdiction(question)
    if jurisdiction:
        jur_matching = [c for c in chunks if _chunk_matches_jurisdiction(c, jurisdiction)]
        if not jur_matching:
            return {
                "answered": False,
                "answer": "",
                "sources": [],
                "_retrieval_gate": "jurisdiction_mismatch",
                "_question_jurisdiction": jurisdiction,
                "_best_score": best,
            }
        # Restrict (not just reorder) to jurisdiction-matching chunks --
        # otherwise a later filter (e.g. by year) could still pull a
        # wrong-jurisdiction chunk back to the front if it happens to
        # satisfy that other criterion, defeating this check.
        chunks = jur_matching

    # Year handling: dense retrieval reliably finds the right *topic* even
    # when the specific year isn't in the corpus (e.g. "Ladli Yojna in
    # 2019-20" still scores high against the real 2022-23/2024-25 documents
    # about the same scheme), but generation only sees a couple of the
    # highest-ranked chunks -- so the right year's chunk can be sitting a
    # few ranks down and never reach the model. If the question names a
    # specific financial year, re-rank the retrieved chunks so any chunk
    # whose text actually contains that year comes first; if genuinely none
    # of them do, that's a reliable signal the corpus doesn't cover this
    # year, so abstain deterministically rather than let a small local LLM
    # guess from the wrong year's figures (observed doing exactly that).
    question_years = _normalize_years(question)
    if question_years:
        matching = [c for c in chunks if question_years & _normalize_years(c["text"])]
        if not matching:
            return {
                "answered": False,
                "answer": "",
                "sources": [],
                "_retrieval_gate": "year_mismatch",
                "_question_years": list(question_years),
                "_best_score": best,
            }
        non_matching = [c for c in chunks if c not in matching]
        chunks = matching + non_matching

    result = llm_answer(question, chunks)
    result["_best_score"] = best
    return result


def to_record(question_id, result):
    sources = []
    for s in result.get("sources", []):
        if isinstance(s, dict) and s.get("file"):
            sources.append({"file": s["file"], "page": s.get("page")})
    return {
        "question_id": question_id,
        "answered": bool(result.get("answered", False)),
        "answer": result.get("answer", "") if result.get("answered") else "",
        "sources": sources,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions_file")
    ap.add_argument("out_file")
    ap.add_argument("--debug", action="store_true", help="also write raw LLM output alongside")
    args = ap.parse_args()

    retriever = Retriever()
    questions = [json.loads(l) for l in open(args.questions_file)]

    out_records = []
    debug_records = []
    for q in questions:
        print(f"[{q['question_id']}] {q['question']}")
        result = run_query(retriever, q["question"])
        record = to_record(q["question_id"], result)
        print(f"  -> answered={record['answered']} answer={record['answer']!r}")
        out_records.append(record)
        debug_records.append({**record, "_debug": result})

    with open(args.out_file, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.debug:
        debug_path = Path(args.out_file).with_suffix(".debug.jsonl")
        with open(debug_path, "w") as f:
            for r in debug_records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"Debug output written to {debug_path}")

    print(f"Wrote {len(out_records)} records to {args.out_file}")


if __name__ == "__main__":
    main()
