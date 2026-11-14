"""Evaluate retrieval methods and context budgets on dev/test splits."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))

from baseline import pack_context  # noqa: E402
from retrievers import (  # noqa: E402
    HybridRRFRetriever,
    LLMRerankRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    chunk_text = norm_text(chunk["text"])
    return any(
        chunk["source_id"] == g["source_id"]
        and chunk["page"] == g["page"]
        and norm_text(g["evidence"]) in chunk_text
        for g in gold
    )


def build_retriever(
    method: str,
    chunks: list[dict[str, Any]],
    *,
    pg_strategy: str,
    candidate_k: int,
):
    if method == "tfidf":
        return TfidfRetriever(chunks)
    if method == "dense":
        return PgVectorDenseRetriever(strategy=pg_strategy)
    if method == "hybrid":
        dense = PgVectorDenseRetriever(strategy=pg_strategy)
        lexical = TfidfRetriever(chunks)
        return HybridRRFRetriever(dense, lexical, candidate_k=candidate_k)
    if method == "dense_rerank":
        dense = PgVectorDenseRetriever(strategy=pg_strategy)
        return LLMRerankRetriever(dense, candidate_k=candidate_k)
    if method == "hybrid_rerank":
        dense = PgVectorDenseRetriever(strategy=pg_strategy)
        lexical = TfidfRetriever(chunks)
        hybrid = HybridRRFRetriever(dense, lexical, candidate_k=max(candidate_k, 12))
        return LLMRerankRetriever(hybrid, candidate_k=candidate_k)
    raise ValueError(f"Unknown method: {method}")


def evaluate_retriever(
    retriever,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 3000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=q.get("filter"), top_k=top_k)
        context, context_ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        marked = [{**r, "hit": is_hit(r, q["gold"])} for r in results]
        top3 = marked[:3]

        traces.append(
            {
                "query_id": q["id"],
                "question": q["question"],
                "method": retriever.name,
                "filter": q.get("filter"),
                "top_k": top_k,
                "budget_chars": budget,
                "results": [
                    {
                        "id": r["id"],
                        "page": r["page"],
                        "score": round(float(r["score"]), 6),
                        "hit": r["hit"],
                    }
                    for r in marked
                ],
                "context_ids": context_ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 2),
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(r["hit"] for r in top3),
                "context_hit": any(r["hit"] and r["id"] in context_ids for r in marked),
            }
        )

    n = len(traces)
    h1 = sum(t["hit1"] for t in traces)
    h3 = sum(t["hit3"] for t in traces)
    ch = sum(t["context_hit"] for t in traces)
    summary = {
        "method": retriever.name,
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "hit1": f"{h1}/{n} ({h1/n:.1%})" if n else "0/0",
        "hit3": f"{h3}/{n} ({h3/n:.1%})" if n else "0/0",
        "context_evidence_rate": f"{ch}/{n} ({ch/n:.1%})" if n else "0/0",
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2) if n else 0,
        "total_cost_usd": round(getattr(retriever, "total_cost_usd", 0.0), 6),
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods and context budget")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--method",
        choices=("tfidf", "dense", "hybrid", "dense_rerank", "hybrid_rerank"),
        default="tfidf",
    )
    parser.add_argument("--pg-strategy", default="B_paragraph")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    eval_qs = [q for q in questions if q.get("split") == args.split and q.get("answerable", True)]

    retriever = build_retriever(
        args.method, chunks, pg_strategy=args.pg_strategy, candidate_k=args.candidate_k
    )
    traces, summary = evaluate_retriever(
        retriever, eval_qs, top_k=args.top_k, budget=args.budget
    )
    summary["split"] = args.split

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
