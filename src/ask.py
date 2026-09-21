"""Ask a single question interactively, for local testing/demoing.

Usage:
    python3 ask.py "What was Delhi's outlay for Ladli Yojna in 2024-25?"
    python3 ask.py              # drops into a REPL, one question per line
"""
import sys

from query import run_query, to_record
from retrieve import Retriever


def ask(retriever, question):
    result = run_query(retriever, question)
    record = to_record("adhoc", result)

    print(f"\nQ: {question}")
    if record["answered"]:
        print(f"A: {record['answer']}")
        for s in record["sources"]:
            page = f", page {s['page']}" if s.get("page") is not None else ""
            print(f"   source: {s['file']}{page}")
    else:
        gate = result.get("_retrieval_gate")
        reason = f" ({gate})" if gate else ""
        print(f"A: [could not answer from the corpus]{reason}")
    print()


def main():
    print("Loading index...", file=sys.stderr)
    retriever = Retriever()

    if len(sys.argv) > 1:
        ask(retriever, " ".join(sys.argv[1:]))
        return

    print("Ready. Type a question and press Enter (Ctrl-D to quit).\n")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        ask(retriever, question)


if __name__ == "__main__":
    main()
