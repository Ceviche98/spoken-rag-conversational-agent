"""
Dense Neural Passage Retrieval & MS MARCO Indexing via Contriever (Lab 4).

Implements:
- In-memory dense vector index for passages with inner-product lookup
- Cosine and dot-product relevance ranking
- Retrieval evaluation metrics: MRR@k, Recall@k, and Precision@k
- Query-passage semantic alignment scoring
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DensePassageRetriever:
    """
    In-memory dense passage retriever based on dual-encoder representations (e.g. Contriever).
    Stores passage IDs, text contents, and normalized embedding vectors.
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        self.embedding_dim = embedding_dim
        self.passage_ids: List[str] = []
        self.passages: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_passages(
        self,
        passage_ids: List[str],
        passages: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Register passages and their corresponding embedding vectors."""
        if len(passage_ids) != len(passages) or len(passage_ids) != len(embeddings):
            raise ValueError("Mismatched counts for passage_ids, passages, and embeddings.")
        self.passage_ids = list(passage_ids)
        self.passages = list(passages)
        # Normalize vectors for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Retrieve top_k passages most similar to query_embedding.
        Returns list of (passage_id, passage_text, score).
        """
        if self.embeddings is None or len(self.passage_ids) == 0:
            return []

        q = query_embedding.flatten()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = np.dot(self.embeddings, q)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                (self.passage_ids[idx], self.passages[idx], float(scores[idx]))
            )
        return results


def compute_mrr_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    """Compute Reciprocal Rank at cutoff k for a single query."""
    for rank, pid in enumerate(ranked_ids[:k], 1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    """Compute Recall at cutoff k for a single query."""
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def evaluate_retriever(
    retriever: DensePassageRetriever,
    queries: List[Dict[str, Any]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate retriever across a list of query dicts containing:
    'id', 'embedding', 'gold_passage_id'.
    """
    mrrs = []
    recalls = []

    for q in queries:
        gold = q["gold_passage_id"]
        results = retriever.retrieve(q["embedding"], top_k=k)
        ranked = [r[0] for r in results]
        mrrs.append(compute_mrr_at_k(ranked, gold, k=k))
        recalls.append(compute_recall_at_k(ranked, gold, k=k))

    return {
        f"MRR@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "num_queries": len(queries),
    }
