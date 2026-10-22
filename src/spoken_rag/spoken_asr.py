"""
Spoken Dialogue ASR Evaluation & Acoustic Noise Robustness (Lab 8).

Implements:
- Word Error Rate (WER) and Character Error Rate (CER) via dynamic programming Levenshtein distance
- Transcript text normalization (lowercasing, punctuation stripping, whitespace normalization)
- Semantic cosine similarity resilience under acoustic transcription errors
- ASR evaluation results loader and statistical model summarizer across SNR noise conditions
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def normalize_transcript(text: str) -> str:
    """Normalize spoken/reference transcripts for strict phonetic and lexical alignment."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using Levenshtein distance over word sequences."""
    r = normalize_transcript(reference).split()
    h = normalize_transcript(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0

    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return float(d[len(r)][len(h)]) / float(len(r))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) using character-level Levenshtein distance."""
    r = normalize_transcript(reference).replace(" ", "")
    h = normalize_transcript(hypothesis).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0

    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return float(d[len(r)][len(h)]) / float(len(r))


def compute_semantic_similarity(reference: str, hypothesis: str) -> float:
    """
    Lightweight token-level Jaccard and length-normalized semantic overlap fallback.
    Returns value in [0.0, 1.0].
    """
    r_words = set(normalize_transcript(reference).split())
    h_words = set(normalize_transcript(hypothesis).split())
    if not r_words and not h_words:
        return 1.0
    if not r_words or not h_words:
        return 0.0
    inter = len(r_words.intersection(h_words))
    union = len(r_words.union(h_words))
    return float(inter) / float(union) if union > 0 else 0.0


def load_asr_evaluation_results(csv_path: str | Path) -> pd.DataFrame:
    """Load empirical ASR benchmarking CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ASR evaluation CSV not found at: {path}")
    return pd.read_csv(path)


def summarize_asr_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Aggregate mean WER, latency, and RTF across model variants in asr_evaluation_results.csv."""
    summary: Dict[str, Dict[str, float]] = {}

    model_mappings = [
        ("Whisper-Base", "Output (Base)", "WER (Base)", "Lat (Base)", "RTF (Base)"),
        ("Distil-Whisper", "Output (Distil)", "WER (Distil)", "Lat (Distil)", "RTF (Distil)"),
        ("Whisper-Turbo", "Output (Turbo)", "WER (Turbo)", "Lat (Turbo)", "RTF (Turbo)"),
    ]

    ref_col = "Normalized Reference"

    for model_name, out_col, wer_col, lat_col, rtf_col in model_mappings:
        if wer_col in df.columns:
            mean_wer = float(df[wer_col].mean())
            mean_lat = float(df[lat_col].mean()) if lat_col in df.columns else 0.0
            mean_rtf = float(df[rtf_col].mean()) if rtf_col in df.columns else 0.0

            # Compute semantic similarities if reference and output are present
            sims = []
            if ref_col in df.columns and out_col in df.columns:
                for _, row in df.iterrows():
                    ref = str(row[ref_col])
                    hyp = str(row[out_col])
                    sims.append(compute_semantic_similarity(ref, hyp))
            mean_sim = float(sum(sims) / len(sims)) if sims else 1.0 - mean_wer

            summary[model_name] = {
                "mean_wer": mean_wer,
                "mean_lat": mean_lat,
                "mean_rtf": mean_rtf,
                "mean_sim": mean_sim,
                "samples": len(df),
            }

    return summary
