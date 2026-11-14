"""Run Agentic RAG on a split (dev or test) and export traces + cases + review template."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import AgenticRAG  # noqa: E402
from eval_retrieval import build_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic RAG on dev or test split")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--method",
        choices=("tfidf", "dense", "hybrid", "dense_rerank", "hybrid_rerank"),
        default="hybrid",
    )
    parser.add_argument("--pg-strategy", default="B_paragraph")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--context-mode", choices=("fifo", "score_desc"), default="score_desc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in questions if q.get("split") == args.split]

    retriever = build_retriever(
        args.method, chunks, pg_strategy=args.pg_strategy, candidate_k=args.candidate_k
    )
    agent = AgenticRAG(
        retriever,
        model=args.model,
        max_searches=args.max_searches,
        top_k=args.top_k,
        budget=args.budget,
        context_mode=args.context_mode,
    )

    traces = []
    cases = []
    reviews_template = []

    for q in selected:
        answerable = bool(q.get("answerable", True))
        trace = agent.run(q["id"], q["question"], filters=q.get("filter"))
        trace["question"] = q["question"]
        trace["answerable"] = answerable
        traces.append(trace)

        cases.append({"id": q["id"], "split": q["split"], "answerable": answerable})

        # Template for manual grading required by starter/agent_metrics.py
        if trace["status"] != "error":
            entry = {
                "query_id": q["id"],
                "question": q["question"],
                "expected_answer": q.get("expected_answer", q.get("absence_reason", "")),
                "agent_status": trace["status"],
                "agent_answer": trace["answer"],
                "citations": trace["citations"],
                "citation_support": [None for _ in trace["citations"]],
                "notes": "",
            }
            if answerable:
                entry["answer_grade"] = None  # fill 0, 1, or 2
            else:
                entry["abstention_correct"] = None  # fill true or false
            reviews_template.append(entry)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "reviews_template.json").write_text(
        json.dumps(reviews_template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "split": args.split,
        "method": args.method,
        "agent_model": agent.model,
        "n_cases": len(traces),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstained": sum(1 for t in traces if t["status"] == "abstain"),
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "avg_searches": round(statistics.mean(t["num_searches"] for t in traces), 2),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
