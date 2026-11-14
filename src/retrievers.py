"""Retrieval adapters for Day 7: Lexical (TF-IDF), Dense (Day 6 pgvector), Hybrid RRF, and LLM Rerank."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from baseline import TfidfIndex  # noqa: E402


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# USD per 1M tokens (approximate standard OpenAI pricing for cost reporting)
MODEL_PRICING_PER_1M = {
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
}


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    in_price, out_price = MODEL_PRICING_PER_1M.get(model, (0.0, 0.0))
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000.0


def _matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for key, expected in filters.items():
        actual = meta.get(key, chunk.get(key))
        if actual != expected:
            return False
    return True


def _format_vector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class RetrieverProtocol(Protocol):
    name: str
    total_cost_usd: float

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        ...


class TfidfRetriever:
    """Lexical TF-IDF retriever using Day 7 starter TfidfIndex with metadata filter support."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.name = "tfidf"
        self.chunks = chunks
        self.index = TfidfIndex(chunks)
        self.total_cost_usd = 0.0

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        # Search full corpus to preserve corpus-wide IDF, then apply metadata filter if provided.
        raw = self.index.search(query, top_k=len(self.chunks) if filters else top_k)
        if filters:
            raw = [c for c in raw if _matches_filter(c, filters)][:top_k]
        return [{**c, "strategy": self.name} for c in raw]


class PgVectorDenseRetriever:
    """Dense retriever querying Day 6 PostgreSQL + pgvector index with OpenAI embeddings."""

    def __init__(
        self,
        *,
        strategy: str = "B_paragraph",
        embedding_model: str | None = None,
        database_url: str | None = None,
        env_path: Path | None = None,
    ) -> None:
        load_dotenv(env_path or (ROOT / ".env"))
        self.database_url = (database_url or os.getenv("DATABASE_URL", "")).strip()
        if not self.database_url:
            raise ValueError("Missing DATABASE_URL in .env or environment")

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env or environment")

        self.embedding_model = (
            embedding_model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        ).strip()
        if self.embedding_model not in EMBEDDING_DIMENSIONS:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

        self.dimensions = EMBEDDING_DIMENSIONS[self.embedding_model]
        self.pg_strategy = strategy
        self.name = "dense"
        self.client = OpenAI(api_key=api_key)
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    def _embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(model=self.embedding_model, input=[query])
        usage_tokens = getattr(response.usage, "total_tokens", 0) or 0
        self.total_tokens += usage_tokens
        self.total_cost_usd += compute_cost_usd(self.embedding_model, usage_tokens)
        return response.data[0].embedding

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        query_vec = self._embed_query(query)
        vec_literal = _format_vector(query_vec)

        params: list[Any] = [vec_literal, self.pg_strategy]
        filter_clause = ""
        if filters:
            filter_clause = "AND metadata @> %s"
            params.append(Jsonb(filters))
        params.extend([vec_literal, top_k])

        sql = f"""
        SELECT
            chunk_id,
            document_id,
            source,
            page,
            text,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        WHERE strategy = %s
        {filter_clause}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
        """
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "source_id": r[1],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": r[5],
                "score": float(r[6]),
                "strategy": self.name,
            }
            for r in rows
        ]


class HybridRRFRetriever:
    """Combines Dense and Lexical (TF-IDF) rankings via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense: PgVectorDenseRetriever,
        lexical: TfidfRetriever,
        *,
        candidate_k: int = 12,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.name = "hybrid_rrf"
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    @property
    def total_cost_usd(self) -> float:
        return self.dense.total_cost_usd + self.lexical.total_cost_usd

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        fetch_k = max(top_k, self.candidate_k)
        dense_hits = self.dense.search(query, filters=filters, top_k=fetch_k)
        lexical_hits = self.lexical.search(query, filters=filters, top_k=fetch_k)

        scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict[str, Any]] = {}

        for rank, chunk in enumerate(dense_hits, start=1):
            cid = chunk["id"]
            chunks_by_id[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank)

        for rank, chunk in enumerate(lexical_hits, start=1):
            cid = chunk["id"]
            chunks_by_id.setdefault(cid, chunk)
            scores[cid] = scores.get(cid, 0.0) + self.lexical_weight / (self.rrf_k + rank)

        ranked_ids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))[:top_k]
        return [
            {**chunks_by_id[cid], "score": scores[cid], "strategy": self.name}
            for cid in ranked_ids
        ]


class LLMRerankRetriever:
    """Two-stage retriever: fetch candidate_k chunks from a base retriever, then rerank with an LLM."""

    def __init__(
        self,
        base_retriever: RetrieverProtocol,
        *,
        candidate_k: int = 10,
        rerank_model: str | None = None,
        env_path: Path | None = None,
    ) -> None:
        load_dotenv(env_path or (ROOT / ".env"))
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env or environment")
        self.base = base_retriever
        self.candidate_k = candidate_k
        self.rerank_model = (rerank_model or os.getenv("AGENT_MODEL") or "gpt-4o-mini").strip()
        self.name = f"{base_retriever.name}_rerank"
        self.client = OpenAI(api_key=api_key)
        self._rerank_cost_usd = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.base.total_cost_usd + self._rerank_cost_usd

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        candidates = self.base.search(query, filters=filters, top_k=max(top_k, self.candidate_k))
        if len(candidates) <= 1:
            return candidates[:top_k]

        cand_map = {c["id"]: c for c in candidates}
        formatted = "\n\n".join(
            f"ID: {c['id']}\nPage: {c['page']}\nText: {c['text'][:700]}"
            for c in candidates
        )
        prompt = (
            "Rank the candidate passages from most relevant to least relevant for answering the question.\n"
            "Return ONLY a JSON object: {\"ranked_ids\": [\"id1\", \"id2\", ...]}.\n\n"
            f"Question: {query}\n\nCandidates:\n{formatted}"
        )
        response = self.client.chat.completions.create(
            model=self.rerank_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a precise passage reranker. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        usage = response.usage
        if usage:
            self._rerank_cost_usd += compute_cost_usd(
                self.rerank_model, usage.prompt_tokens or 0, usage.completion_tokens or 0
            )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        ranked_ids = [cid for cid in parsed.get("ranked_ids", []) if cid in cand_map]

        # Append any valid candidate IDs omitted by the model in original order
        seen = set(ranked_ids)
        for c in candidates:
            if c["id"] not in seen:
                ranked_ids.append(c["id"])
                seen.add(c["id"])

        return [
            {
                **cand_map[cid],
                "score": 1.0 / rank,
                "strategy": self.name,
            }
            for rank, cid in enumerate(ranked_ids[:top_k], start=1)
        ]
