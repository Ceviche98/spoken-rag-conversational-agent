"""
Spoken RAG Conversational Agent: Whisper ASR + Contriever MS MARCO Retrieval.
"""

from __future__ import annotations

from spoken_rag.agent import SpokenRAGAgent
from spoken_rag.retriever import (
    DensePassageRetriever,
    compute_mrr_at_k,
    compute_recall_at_k,
    evaluate_retriever,
)
from spoken_rag.spoken_asr import (
    compute_cer,
    compute_semantic_similarity,
    compute_wer,
    load_asr_evaluation_results,
    normalize_transcript,
    summarize_asr_models,
)

__all__ = [
    "SpokenRAGAgent",
    "DensePassageRetriever",
    "compute_wer",
    "compute_cer",
    "compute_semantic_similarity",
    "normalize_transcript",
    "load_asr_evaluation_results",
    "summarize_asr_models",
    "compute_mrr_at_k",
    "compute_recall_at_k",
    "evaluate_retriever",
]
