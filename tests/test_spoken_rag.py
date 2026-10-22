"""Unit tests for the spoken_rag package."""

from __future__ import annotations

from pathlib import Path
import numpy as np

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


def test_spoken_asr_metrics() -> None:
    assert normalize_transcript("Hello, World! 123") == "hello world 123"
    assert compute_wer("dense vectors", "dense vectors") == 0.0
    assert compute_wer("dense vectors", "dense embeddings") == 0.5
    assert compute_cer("word", "word") == 0.0
    assert compute_semantic_similarity("hello world", "hello there") > 0.0


def test_asr_csv_loader() -> None:
    csv_path = Path(__file__).parent.parent / "data" / "asr_evaluation_results.csv"
    assert csv_path.exists()
    df = load_asr_evaluation_results(csv_path)
    assert len(df) > 0
    summary = summarize_asr_models(df)
    assert "Whisper-Base" in summary
    assert "Whisper-Turbo" in summary
    assert summary["Whisper-Base"]["mean_wer"] >= 0.0


def test_dense_passage_retriever_and_metrics() -> None:
    dim = 16
    retriever = DensePassageRetriever(embedding_dim=dim)
    p_ids = ["p1", "p2", "p3"]
    texts = [
        "Quantum computing relies on quantum superposition.",
        "Deep learning architectures use backpropagation.",
        "Natural language processing processes textual data.",
    ]
    # Synthetic orthogonal-ish embeddings
    embeds = np.zeros((3, dim), dtype=np.float32)
    embeds[0, 0] = 1.0
    embeds[1, 1] = 1.0
    embeds[2, 2] = 1.0

    retriever.add_passages(p_ids, texts, embeds)

    # Query matching p1
    q1 = np.zeros(dim, dtype=np.float32)
    q1[0] = 1.0
    results = retriever.retrieve(q1, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "p1"
    assert "Quantum computing" in results[0][1]

    # Metrics
    ranked = ["p1", "p2", "p3"]
    assert compute_mrr_at_k(ranked, "p1", k=10) == 1.0
    assert compute_mrr_at_k(ranked, "p2", k=10) == 0.5
    assert compute_recall_at_k(ranked, "p3", k=3) == 1.0
    assert compute_recall_at_k(ranked, "p4", k=3) == 0.0


def test_spoken_rag_agent_e2e() -> None:
    dim = 8
    retriever = DensePassageRetriever(embedding_dim=dim)
    retriever.add_passages(
        ["doc1"],
        ["Photosynthesis converts sunlight into chemical energy."],
        np.ones((1, dim), dtype=np.float32),
    )

    agent = SpokenRAGAgent(retriever=retriever)
    query_vec = np.ones(dim, dtype=np.float32)

    res = agent.process_spoken_utterance(
        transcription="What is photosynthesis?",
        query_embedding=query_vec,
        top_k=1,
    )

    assert "Photosynthesis converts sunlight" in res["response_text"]
    assert len(res["retrieved_passages"]) == 1
    assert len(agent.dialogue_history) == 1
    agent.clear_history()
    assert len(agent.dialogue_history) == 0
