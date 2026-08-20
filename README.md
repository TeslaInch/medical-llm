# Medical LLM for Sickle Cell Disease 🩸

An end-to-end Machine Learning pipeline and RAG (Retrieval-Augmented Generation) system specifically engineered for clinical decision support in Sickle Cell Disease (SCD) management.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TeslaInch/SCD-Medical-LLM)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)

---

## 🌟 System Architecture

This project has evolved from a simple fine-tuning script into a full production-grade MLOps system.

### 1. The Model (Phi-3.5-mini + LoRA)
We fine-tuned the **3.8B parameter Phi-3.5-mini** model using QLoRA on a highly curated, 6-layer custom medical dataset. The resulting adapter weights were merged, and the model was quantized to an ultra-efficient 8-bit `GGUF` format for rapid CPU inference.
- **Model Checkpoint:** [TeslaInch/phi-3.5-mini-SCD-gguf](https://huggingface.co/TeslaInch/phi-3.5-mini-SCD-gguf)

### 2. Retrieval-Augmented Generation (RAG)
To prevent hallucinations (such as incorrectly diagnosing "Aplastic Episode" in high-fever cases), we integrated a persistent **ChromaDB** vector database containing established clinical guidelines (e.g., ASH Guidelines).
- **Embedder:** `BAAI/bge-small-en-v1.5`
- **Reranker (Cross-Encoder):** `BAAI/bge-reranker-base` (with dynamic thresholding to strip out distractors).
- **Vector DB Dataset:** [TeslaInch/SCD-vectorDB-v1](https://huggingface.co/datasets/TeslaInch/SCD-vectorDB-v1)

### 3. MLOps & Observability
- **FastAPI Inference Server:** A robust API that wraps the `llama.cpp` engine, handles JSON validation via Pydantic, and embeds guardrails.
- **MLflow Tracking:** Every prediction, latency metric, and chunk retrieval is logged for continuous drift monitoring.
- **CI/CD:** Automated testing and deployment to HuggingFace Spaces via GitHub Actions.

---

## 🚀 Live API Usage

The inference server is live and hosted on a HuggingFace Docker Space!

**Endpoint:** `POST https://TeslaInch-SCD-Medical-LLM.hf.space/predict`

```bash
curl -X 'POST' \
  'https://TeslaInch-SCD-Medical-LLM.hf.space/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is the most likely diagnosis for a 4-year-old with HbSS presenting with fever and swollen hands/feet, and what is the immediate management?",
  "case": "Sickle cell anemia patient triage"
}'
```

---

## 📂 Project Structure

```
├── api/                  # FastAPI inference server, Dockerfile, and unit tests
├── data/                 # Raw clinical notes, QA pairs, and VectorDB chunks
├── frontend/             # [Coming Soon] React/Vite Chat Interface
├── scripts/              # Data extraction, RAG evaluation, and deployment scripts
├── training/             # Kaggle notebooks for QLoRA fine-tuning and ablation studies
└── README.md             # This file
```

---

## 📊 Evaluation Benchmarks

Our custom evaluation benchmark tests the model against strict multi-turn clinical conversations and ASH guidelines.

- **Eval Dataset:** [TeslaInch/SCD-Eval-Benchmark](https://huggingface.co/datasets/TeslaInch/SCD-Eval-Benchmark)
- **Training Dataset:** [TeslaInch/SCD-Instruction-Tuning](https://huggingface.co/datasets/TeslaInch/SCD-Instruction-Tuning)

| Layer | Focus Area | Question Count | Status |
|---|---|---|---|
| 1 | Custom Clinical Notes (Factual Recall) | 50 | ✅ Done |
| 2 | MCQ Benchmark (General Knowledge) | ~400 | ✅ Done |
| 3 | Combined Reasoning (Complex Management) | 40 | ✅ Done |
| 4 | Clinical Cases (Diagnosis & Treatment) | 30 | ✅ Done |
| 5 | ASH Guidelines (Strict Protocol Adherence) | 64 | ✅ Done |
| 6 | Multi-turn clinical conversations | 20 | ✅ Done |
