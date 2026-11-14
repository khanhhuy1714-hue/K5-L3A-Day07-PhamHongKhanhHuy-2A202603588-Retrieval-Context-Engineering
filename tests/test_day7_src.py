"""Unit tests for custom Day 7 retrievers, hit normalization, and context selection."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import select_context  # noqa: E402
from eval_retrieval import is_hit  # noqa: E402
from retrievers import HybridRRFRetriever, TfidfRetriever  # noqa: E402


class _FakeDense:
    def __init__(self, chunks):
        self.name = "dense"
        self.total_cost_usd = 0.0
        self._chunks = chunks

    def search(self, query, *, filters=None, top_k=3):
        return [{**c, "score": 0.9 - idx * 0.1, "strategy": "dense"} for idx, c in enumerate(self._chunks[:top_k])]


class Day7SrcTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "c1",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 20,
                "text": "An algorithm is a sequence\nof precise instructions.",
                "metadata": {"section": "1.1"},
            },
            {
                "id": "c2",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 21,
                "text": "Ada Lovelace was the first programmer.",
                "metadata": {"section": "1.2"},
            },
        ]

    def test_is_hit_normalizes_newlines(self):
        gold = [{"source_id": "doc", "page": 20, "evidence": "sequence of precise instructions"}]
        self.assertTrue(is_hit(self.chunks[0], gold))
        self.assertFalse(is_hit(self.chunks[1], gold))

    def test_tfidf_metadata_filter(self):
        retriever = TfidfRetriever(self.chunks)
        hits = retriever.search("algorithm", filters={"section": "1.2"}, top_k=3)
        self.assertEqual(hits, [])
        hits_ok = retriever.search("algorithm", filters={"section": "1.1"}, top_k=3)
        self.assertEqual([h["id"] for h in hits_ok], ["c1"])

    def test_hybrid_rrf_fuses_ranks(self):
        lexical = TfidfRetriever(self.chunks)
        dense = _FakeDense(list(reversed(self.chunks)))
        hybrid = HybridRRFRetriever(dense, lexical, candidate_k=2)
        hits = hybrid.search("algorithm instructions", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(h["strategy"] == "hybrid_rrf" for h in hits))

    def test_select_context_score_desc(self):
        c_low = {**self.chunks[0], "score": 0.2}
        c_high = {**self.chunks[1], "score": 0.9}
        ctx, ids = select_context([c_low, c_high], budget=120, mode="score_desc")
        self.assertEqual(ids, ["c2"])
        self.assertIn("Ada Lovelace", ctx)


if __name__ == "__main__":
    unittest.main()
