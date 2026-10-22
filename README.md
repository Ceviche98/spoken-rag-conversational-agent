# Spoken RAG Conversational Agent

[![CI](https://github.com/Ceviche98/spoken-rag-conversational-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Ceviche98/spoken-rag-conversational-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Whisper_ASR_%2B_Contriever_RAG-informational.svg)](https://github.com/Ceviche98/spoken-rag-conversational-agent)
[![ASR Benchmark](https://img.shields.io/badge/Whisper--Turbo-0.038_RTF-success.svg)](https://github.com/Ceviche98/spoken-rag-conversational-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Spoken Retrieval-Augmented Generation (Spoken RAG) conversational system designed for voice-driven question answering over unstructured knowledge bases.

The system combines **OpenAI Whisper Automatic Speech Recognition (ASR)** for robust spoken transcription under acoustic noise with **Contriever dense neural passage retrieval** over the MS MARCO corpus, synthesizing factually grounded responses paired with Text-to-Speech (TTS) audio output.

---

## 1. System Architecture

```
User Voice Input
       │
       ▼
[1. Speech-to-Text] ──► OpenAI Whisper ASR (Base / Distil / Turbo)
       │
       ▼ Cleaned Spoken Transcript
[2. Dense Retrieval] ──► Contriever Dual-Encoder (768-D MS MARCO Passage Index)
       │
       ▼ Top-K Grounded Passages
[3. Answer Synthesis] ──► Context-Conditioned Neural Generator
       │
       ▼ Synthesized Response Text
[4. Voice Output] ──► Text-to-Speech (TTS) Audio Engine
```

### Core Components
1. **Acoustic Transcription Layer (`src/spoken_rag/spoken_asr.py`):**
   - Transcribes user speech across Whisper variants (`Whisper-Base`, `Distil-Whisper`, `Whisper-Turbo`).
   - Normalizes surface tokens (case folding, punctuation stripping, whitespace normalization).
   - Evaluates Word Error Rate (WER), Character Error Rate (CER), and semantic embedding similarity under ambient acoustic noise.
2. **Dense Passage Retrieval Layer (`src/spoken_rag/retriever.py`):**
   - Dense bi-encoder vector indexing over MS MARCO passage collections.
   - Fast inner-product dot-product search with L2 normalization for exact cosine similarity ranking.
   - Evaluated on Mean Reciprocal Rank (MRR@k) and Recall@k.
3. **Conversational Agent Orchestrator (`src/spoken_rag/agent.py`):**
   - Coordinates the multi-turn conversational loop, attaches retrieved passages as factual grounding, and prepares synthesized outputs for audio playback.

---

## 2. Empirical ASR Benchmarks & Acoustic Noise Robustness

Evaluated across canonical speech evaluation benchmarks under variable signal-to-noise ratios (SNR):

<div align="center">
  <img src="docs/assets/spoken_rag_benchmarks.png" width="95%" alt="Spoken RAG Benchmarks: ASR Accuracy, Latency, and Semantic Alignment" />
  <p><em><strong>Figure 1:</strong> (Left) Whisper ASR variant comparison across Word Error Rate and inference latency. (Right) Downstream semantic alignment as a function of transcription error.</em></p>
</div>

### Whisper Model Comparison (`data/asr_evaluation_results.csv`)

| Model Architecture | Parameter Count | Mean WER (%) | Latency (s) | Real-Time Factor (RTF) | Downstream Semantic Alignment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Whisper-Turbo** | 809M | **1.82%** | **0.284s** | **0.038** | **98.4%** |
| **Distil-Whisper** | 756M | 2.14% | 0.312s | 0.042 | 97.9% |
| **Whisper-Base** | 74M | 4.86% | 0.445s | 0.059 | 94.2% |

### Key Findings
- **Real-Time Factor (RTF):** Whisper-Turbo delivers sub-second inference latency ($0.284\text{s}$, $\text{RTF} = 0.038$), processing audio over 25x faster than real-time.
- **Acoustic Invariance:** While small transcription slips occur in domain-specific technical terminology, dense neural retrievers (Contriever) preserve high semantic alignment ($>98\%$), effectively insulating the downstream QA generator from phonetic errors.

---

## 3. Repository Structure

```
spoken-rag-conversational-agent/
├── data/
│   └── asr_evaluation_results.csv      # Empirical ASR evaluation benchmark dataset
├── docs/
│   └── assets/
│       └── spoken_rag_benchmarks.png   # Benchmark plots (WER, latency, semantic alignment)
├── scripts/
│   └── evaluate_spoken_rag.py          # Benchmark evaluation and plotting utility
├── src/
│   └── spoken_rag/
│       ├── __init__.py
│       ├── agent.py                    # End-to-end Spoken RAG conversational pipeline
│       ├── retriever.py                # Contriever dense vector index and passage search
│       └── spoken_asr.py               # Whisper ASR transcription, WER/CER metrics & normalization
├── tests/
│   └── test_spoken_rag.py              # Automated unit and integration test suite
├── .gitignore
├── LICENSE                             # MIT License
├── pyproject.toml
└── requirements.txt
```

---

## 4. Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Ceviche98/spoken-rag-conversational-agent.git
cd spoken-rag-conversational-agent
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Benchmark Suite
```bash
python scripts/evaluate_spoken_rag.py
```

### 3. Run the Unit Tests
```bash
pytest tests/ -v
```

---

## License

This project is licensed under the MIT License: see the [LICENSE](LICENSE) file for details.
