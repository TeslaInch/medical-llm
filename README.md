# Medical LLM for Sickle Cell Disease

This repository contains the codebase and data pipelines for evaluating and fine-tuning an LLM specifically tailored for Sickle Cell Disease (SCD) management.

## Status

- [x] Training dataset: 203 examples (Layer 1 + 3 + 4 + medalpaca filtered), Alpaca format
- [x] Eval benchmark: 6 layers including multi-turn clinical conversations
- [ ] Fine-tuning: Phi-3 Mini 3.8B with QLoRA (in process)

## Evaluation Benchmark

The custom evaluation benchmark is publicly available on HuggingFace: [TeslaInch/SCD-Eval-Benchmark](https://huggingface.co/datasets/TeslaInch/SCD-Eval-Benchmark)

| Layer | Focus Area | Question Count | Status |
|---|---|---|---|
| 1 | Custom Clinical Notes (Factual Recall) | 50 | Done |
| 2 | MCQ Benchmark (General Knowledge) | ~400 | Done |
| 3 | Combined Reasoning (Complex Management) | 40 | Done |
| 4 | Clinical Cases (Diagnosis & Treatment) | 30 | Done |
| 5 | ASH Guidelines (Strict Protocol Adherence) | 64 | Done |
| 6 | Multi-turn clinical conversations | 20 | Done |

## Training Dataset

The custom instruction-tuning dataset is publicly available on HuggingFace: [TeslaInch/SCD-Instruction-Tuning](https://huggingface.co/datasets/TeslaInch/SCD-Instruction-Tuning)
