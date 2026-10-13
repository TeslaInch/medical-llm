# Medical LLM for Sickle Cell Disease 🩸

An experimental Machine Learning pipeline and RAG (Retrieval-Augmented Generation) research prototype for clinical decision support exploration in Sickle Cell Disease (SCD) management.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TeslaInch/SCD-Medical-LLM)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)

> [!CAUTION]
> ### ⚠️ EXPERIMENTAL RESEARCH ONLY — NOT FOR UNREVIEWED CLINICAL USE
> This project is an **experimental research prototype** and is **NOT approved or ready for direct hospital, emergency room, or standalone clinical use**.
>
> - **Mandatory Medical Review:** All generated outputs, treatment rationales, and triage recommendations must be thoroughly reviewed and verified by certified healthcare professionals (physicians/hematologists) prior to any clinical consideration.
> - **No Autonomous Decision-Making:** This system must never replace professional clinical judgment or established hospital care pathways.
> - **Regulatory Status:** This system has not been cleared, approved, or certified by any health authority or regulatory body (e.g., FDA, CE-IVD/SaMD).

---

## 🌟 System Architecture

This project demonstrates an end-to-end MLOps and research architecture for domain-specific medical AI exploration.

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

---

## ⚠️ Clinical Disclaimer & Research Status

> **CRITICAL NOTICE:** This repository is intended strictly for academic, technical, and educational research into Small Language Model (SLM) adaptation and retrieval architectures.

1. **Experimental Prototype:** The models, retrieval pipelines, and APIs in this project are experimental. They have not undergone formal clinical trials, institutional review board (IRB) clinical testing, or prospective validation.
2. **Not for Autonomous Hospital Deployment:** The system must **never** be used as an autonomous diagnostic or prescriptive system in hospital wards, intensive care units, or emergency departments.
3. **Risk of Hallucinations and Inaccuracies:** Although domain fine-tuning and cross-encoder RAG substantially mitigate factual drift, large language models can generate hallucinations, out-of-context advice, or incomplete contraindication checks.
4. **Human-in-the-Loop Validation Required:** Any deployment in a simulated or pilot healthcare setting requires a strict human-in-the-loop workflow where licensed hematologists or physicians inspect, validate, and sign off on all outputs against official clinical protocols.

