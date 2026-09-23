"""Top-level query pipeline: retrieve -> generate -> abstain-gate -> answers.jsonl record."""
import argparse
import json
import re
import sys
from pathlib import Path

from generate import answer as llm_answer
from retrieve import Retriever, _is_devanagari_query

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


# This corpus covers exactly four jurisdictions: three states/UTs plus the
# Union (national) government, which is split across two files (the MRF-13
# CSV and Statement 20 XLS -- there's no single "Union" filename prefix to
# match, unlike the state PDFs). If a question names one of these four
# (English or Hindi script), the answering chunk must come from that
# jurisdiction's own file(s) -- cross-lingual dense retrieval was observed
# to occasionally match a topically-similar page from a *different*
# jurisdiction's document (e.g. a Union Gender Budget Statement question
# retrieving Odisha-document pages that happen to share vocabulary), which
# the small local LLM then answered from without noticing the mismatch.
JURISDICTION_KEYWORDS = {
    "delhi": "delhi", "दिल्ली": "delhi",
    "odisha": "odisha", "orissa": "odisha", "ओड़िशा": "odisha", "उड़ीसा": "odisha",
    "bihar": "bihar", "बिहार": "bihar",
    "union": "union", "भारत सरकार": "union", "केंद्र सरकार": "union",
}
JURISDICTION_FILES = {
    "delhi": ["gender_budget_"],  # prefix match: gender_budget_2022-23.pdf etc.
    "odisha": ["14-Gender_Budget.pdf"],
    "bihar": ["17107782611749468962.pdf"],
    "union": ["MRF_13_Union_Budget.csv", "stat20.xls"],
}


def _question_jurisdiction(question):
    q_lower = question.lower()
    for kw, jurisdiction in JURISDICTION_KEYWORDS.items():
        if kw in q_lower or kw in question:
            return jurisdiction
    return None


def _chunk_matches_jurisdiction(chunk, jurisdiction):
    return any(chunk["file"].startswith(prefix) for prefix in JURISDICTION_FILES[jurisdiction])


# The Union CSV lists many schemes under BOTH Part A and Part B with
# different figures for each -- a legitimate "two right answers" case (see
# questions_dev.jsonl's own note on this). The LLM was observed to see both
# rows in its context and still only report one, silently dropping the
# other rather than synthesizing "both" reliably. Since the CSV's row
# format is fixed and machine-generated (src/ingest.py's extract_union_csv),
# it's more reliable to detect this pattern and build the answer directly
# from the parsed fields than to keep hoping the LLM notices.
_CSV_ROW_RE = re.compile(r"Category:\s*(PART [AB])[^|]*\|.*?Scheme:\s*([^|]+?)\s*\|")
_CSV_YEAR_VALUE_RE = re.compile(r"(\d{4}-\d{4} Budget Estimates):\s*([\d.]+)")


def _try_dual_part_csv_answer(chunks, question_years):
    by_scheme = {}
    for c in chunks:
        if not c["file"].endswith(".csv"):
            continue
        m = _CSV_ROW_RE.search(c["text"])
        if not m:
            continue
        part, scheme = m.group(1), m.group(2).strip().lower()
        by_scheme.setdefault(scheme, {})[part] = c

    for parts in by_scheme.values():
        if "PART A" not in parts or "PART B" not in parts:
            continue

        target_col = None
        if question_years:
            start = next(iter(question_years)).split("-")[0]
            target_col = f"{start}-{int(start) + 1} Budget Estimates"

        values = {}
        for label, c in parts.items():
            col_values = dict(_CSV_YEAR_VALUE_RE.findall(c["text"]))
            if target_col and target_col in col_values:
                values[label] = col_values[target_col]
            elif col_values:
                # fall back to the last (rightmost / most recent) column
                values[label] = list(col_values.values())[-1]

        if values.get("PART A") and values.get("PART B"):
            answer = (
                f"Under Part A it is Rs {values['PART A']} crore; "
                f"under Part B it is Rs {values['PART B']} crore."
            )
            return {
                "answered": True,
                "answer": answer,
                "sources": [
                    {"file": parts["PART A"]["file"], "page": None},
                    {"file": parts["PART B"]["file"], "page": None},
                ],
                "_deterministic_dual_part": True,
            }
    return None

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
# Retrieve a fairly wide pool -- only 1 page (or a few short CSV/XLS rows)
# actually reaches the model (see generate.build_context), but the year/jurisdiction
# guardrails below need enough candidates to find a same-year,
# same-jurisdiction chunk even when it isn't in, say, the top 6 by raw
# similarity (observed: the correct Delhi 2022-23 page ranked ~12th for one
# Hindi-language query, behind several higher-scoring but wrong-year/
# wrong-jurisdiction pages).
TOP_K = 15


SINGLE_FILE_JURISDICTIONS = {"bihar", "odisha"}


def run_query(retriever, question, top_k=TOP_K):
    # The retriever caps results per file so one long document can't crowd
    # out the others. When the question is scoped to a jurisdiction that IS
    # a single file, there's nothing to stay diverse against, and the cap
    # just throws candidates away (observed: the Odisha summary page with
    # the answer never entered the pool because 3 other Odisha pages
    # outranked it on whole-page similarity).
    jurisdiction = _question_jurisdiction(question)
    per_file = top_k if jurisdiction in SINGLE_FILE_JURISDICTIONS else 3
    chunks = retriever.search(question, top_k=top_k, max_per_file=per_file)
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
    #
    # Exception: Bihar and Odisha are each a single document covering one
    # year only (unlike the Delhi PDFs, which repeat their year on every
    # page, or the Union CSV/XLS, which mix years in the same table). Their
    # pages don't restate "2024-25" on every page, so requiring it in-chunk
    # produced false abstentions on genuinely-correct pages (observed on a
    # Bihar page with the right nutrition-rate figures but no literal year
    # string). Jurisdiction match already guarantees the year is right for
    # these two, so skip the per-chunk year check.
    SINGLE_YEAR_JURISDICTIONS = {"bihar", "odisha"}
    question_years = _normalize_years(question)
    priority_len = len(chunks)
    if question_years and jurisdiction not in SINGLE_YEAR_JURISDICTIONS:
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
        # A page merely *mentioning* the year isn't the same as a document
        # *for* that year: the 2025-26 Delhi PDF lists 2024-25 as its
        # Revised Estimates column, so it passes the check above for a
        # "2024-25" question too (observed: answered from it, and got the
        # 2025-26 outlay). Prefer documents whose own filename year matches.
        own_year = [
            c for c in matching
            if question_years & _normalize_years(c["file"].replace("_", " "))
        ]
        if own_year:
            matching = own_year + [c for c in matching if c not in own_year]
        non_matching = [c for c in chunks if c not in matching]
        chunks = matching + non_matching
        priority_len = len(own_year) if own_year else len(matching)

    # When the top candidate is a full PDF page, generation only gets that
    # one page -- so which page is first matters a lot. Re-rank the top few
    # (within the group that already passed the year/jurisdiction checks,
    # so this can never undo them) by best-matching passage rather than
    # whole-page similarity.
    # Skipped for Hindi-script questions: a 3-line window is too little
    # context for cross-lingual matching, and it demoted the correct page
    # on two Hindi questions that whole-page similarity had gotten right.
    RERANK_POOL = 6
    if chunks and len(chunks[0]["text"]) >= 600 and not _is_devanagari_query(question):
        head = min(priority_len, RERANK_POOL)
        chunks = retriever.rerank_by_passage(question, chunks[:head]) + chunks[head:]

    if jurisdiction == "union":
        dual = _try_dual_part_csv_answer(chunks, question_years)
        if dual:
            dual["_best_score"] = best
            return dual

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
    # Windows' default console/file encoding (cp1252) can't print the Hindi
    # questions; force UTF-8 so a redirected run doesn't crash.
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("questions_file")
    ap.add_argument("out_file")
    ap.add_argument("--debug", action="store_true", help="also write raw LLM output alongside")
    args = ap.parse_args()

    retriever = Retriever()
    questions = [json.loads(l) for l in open(args.questions_file, encoding="utf-8")]

    out_records = []
    debug_records = []
    for q in questions:
        print(f"[{q['question_id']}] {q['question']}")
        result = run_query(retriever, q["question"])
        record = to_record(q["question_id"], result)
        print(f"  -> answered={record['answered']} answer={record['answer']!r}")
        out_records.append(record)
        debug_records.append({**record, "_debug": result})

    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.debug:
        debug_path = Path(args.out_file).with_suffix(".debug.jsonl")
        with open(debug_path, "w", encoding="utf-8") as f:
            for r in debug_records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"Debug output written to {debug_path}")

    print(f"Wrote {len(out_records)} records to {args.out_file}")


if __name__ == "__main__":
    main()
