
# Salience-Guided Prompt Compression for Abstractive Summarization

<p align="center">
  <img src="https://img.shields.io/badge/Python_3.x-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

This repository contains two extensions for efficient LLM-based abstractive summarization using **SigExt** keyphrase salience scores. The goal is to reduce prompt length — and therefore API cost and latency — while preserving summary quality. Both modules are evaluated using ROUGE and BERTScore against full-text baselines.

> Full methodology and experimental results: [Report.pdf](Report.pdf)

---

## Overview

Prompting LLMs with full documents is expensive and can exceed context-window limits. **SigExt** is a Longformer-based extractor that assigns salience scores to keyphrases identified in a document. This project leverages those scores to build two practical prompt-reduction strategies:

1. **SigCompressor** — Single-document prompt compression on CNN/DailyMail, operating at sentence or phrase granularity.
2. **SigSelector** — Multi-document source selection on Multi-News, ranking and filtering entire documents by aggregated keyphrase salience before summarization.

Both modules are model-agnostic: the same SigExt scoring layer is reused across GPT-3.5-Turbo, Mistral-7B, and Llama-3.1-8B.

---

## Repository Structure

```
├── SigCompressor/               # Extension 1: Single-document compression
│   ├── src/
│   │   ├── compressors.py       # Sentence-level & phrase-level compressors
│   │   ├── pipeline.py          # Compression → generation → evaluation pipeline
│   │   ├── benchmark.py         # Full-text baseline runner
│   │   ├── metrics.py           # ROUGE, BERTScore, BLEU computation
│   │   └── utils.py             # Shared utilities
│   ├── prepare_sigext.py        # Clone SigExt, train on CNN/DailyMail, run inference
│   ├── main.py                  # Experiment entry point
│   ├── analyze_results.py       # Plot compression vs. metrics trade-offs
│   └── requirements.txt
│
├── SigSelector/                 # Extension 2: Multi-document selection
│   ├── src/
│   │   ├── selector.py          # Document scoring & top-k selection
│   │   └── benchmark.py        # Full-text vs. selected-text benchmark
│   ├── prepare_sigext.py        # Clone SigExt, train on Multi-News, run inference
│   ├── main.py                  # Experiment entry point
│   └── requirements.txt
│
└── Report.pdf
```

---

## Method

### Extension 1 — SigCompressor

Two compression strategies are applied to CNN/DailyMail articles before summarization:

**Sentence-level compression:** Each sentence is scored by max-pooling the SigExt salience of the keyphrases it contains. The top-`r` fraction of sentences are kept and reordered by their original position. This preserves grammatical fluency and discourse structure at the cost of coarser token control.

**Phrase-level compression:** The top-`r` fraction of keyphrases are selected directly by salience rank and concatenated in document order. This yields the tightest token savings but produces syntactically fragmented prompts, requiring stronger instruction-following from the downstream model.

Both strategies are evaluated at multiple keep ratios `r` targeting 60–80% input reduction, across GPT-3.5-Turbo and Mistral-7B-Instruct-v0.3 (4-bit quantized).

### Extension 2 — SigSelector

For Multi-News clusters of related articles, each document is scored by aggregating the exponentiated SigExt log-probability scores of its keyphrases, normalized by phrase count to avoid length bias:

```
score(D_i) = Σ exp(score(k_j)) / |{k_j ∈ D_i}|
```

The top-k scoring documents are then concatenated and passed to the LLM. We compare k ∈ {1, 2} against full-text prompting using GPT-3.5-Turbo and Llama-3.1-8B-Instant (via Groq).

---

## Results

### Extension 1 — CNN/DailyMail Compression (200 samples)

**GPT-3.5-Turbo**

| Config | Input Reduction | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|--------|:--------------:|:-------:|:-------:|:-------:|:---------:|
| Full Text | 0% | 39.33 | 14.96 | 25.02 | 27.67 |
| Sentence r=0.18 | 79.3% | 37.92 | 13.99 | 24.16 | 26.52 |
| Phrase r=0.30 | **79.3%** | 38.90 | 13.73 | 24.12 | 26.91 |
| Sentence r=0.33 | 61.7% | 39.72 | 15.00 | 25.65 | 27.94 |

**Mistral-7B**

| Config | Input Reduction | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|--------|:--------------:|:-------:|:-------:|:-------:|:---------:|
| Full Text | 0% | 37.15 | 13.15 | 23.42 | 23.70 |
| Sentence r=0.18 | 79.3% | 34.70 | 11.84 | 22.05 | 21.85 |
| Phrase r=0.30 | 79.3% | 36.17 | 11.19 | 21.48 | 21.41 |
| Sentence r=0.33 | 61.7% | **36.99** | **12.70** | **22.53** | **23.46** |

GPT-3.5 is robust to aggressive compression (~79%), with minimal ROUGE degradation. Mistral benefits more from sentence-level prompts and lower reduction ratios (~60%), where enough discourse structure is retained to support coherent generation.

---

### Extension 2 — Multi-News Document Selection (200 samples)

| Model | Strategy | Input Reduction | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|-------|----------|:--------------:|:-------:|:-------:|:-------:|:---------:|
| GPT-3.5-Turbo | Full Text | 0% | 40.73 | 11.59 | 19.63 | 18.40 |
| GPT-3.5-Turbo | **Top-2** | **28.06%** | **40.55** | **11.47** | **19.75** | **17.86** |
| GPT-3.5-Turbo | Top-1 | 64.16% | 37.66 | 10.38 | 18.84 | 15.32 |
| Llama-3.1-8B | Full Text | 0% | 39.15 | 12.77 | 18.61 | 12.01 |
| Llama-3.1-8B | **Top-2** | **28.06%** | **38.68** | **12.44** | **18.34** | **11.02** |
| Llama-3.1-8B | Top-1 | 64.16% | 36.95 | 10.58 | 17.62 | 7.56 |

Top-2 selection achieves a **28% prompt reduction with near-zero quality loss** for both models. Top-1 pushes reduction to 64% but discards cross-source evidence, causing a sharp BERTScore drop, especially for Llama.

---

## Setup & Usage

### SigCompressor

```bash
cd SigCompressor
pip install -r requirements.txt

# 1. Prepare SigExt extractor and CNN/DailyMail dataset
python prepare_sigext.py

# 2. Run compression + summarization benchmark
python main.py --input_file cnn_dataset_with_keyphrase/test.jsonl --openai_key YOUR_KEY

# Advanced: specify compressor, ratios, and LLM
python main.py \
  --input_file cnn_dataset_with_keyphrase/test.jsonl \
  --openai_key YOUR_KEY \
  --compressor phrase \       # sentence | phrase | all
  --ratios 0.2 0.4 \
  --llm mistral               # gpt | mistral | all

# 3. Analyze and visualize results
python analyze_results.py --results_dir results
```

### SigSelector

```bash
cd SigSelector
pip install -r requirements.txt

# 1. Prepare SigExt extractor and Multi-News dataset
python prepare_sigext.py

# 2. Run benchmark (requires API keys)
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
python main.py

# Advanced: specify k and models
python main.py --k 1 2 --models llama   # llama | gpt | all
```

### Output Files

| Module | File | Contents |
|--------|------|----------|
| SigCompressor | `results/*_summaries.json` | Generated summaries per config |
| SigCompressor | `results/master_benchmark_results.csv` | Aggregated ROUGE/BERTScore |
| SigCompressor | `results/metrics_comparison.png` | Trade-off visualization |
| SigSelector | `sigselector_generations.csv` | Raw summaries per strategy |
| SigSelector | `final_benchmark_metrics.csv` | ROUGE/BERTScore per config |

---

## Models Used

| Model | Role | Access |
|-------|------|--------|
| SigExt (Longformer) | Keyphrase salience extraction | HuggingFace |
| GPT-3.5-Turbo | Summarization (both extensions) | OpenAI API |
| Mistral-7B-Instruct-v0.3 | Summarization (Ext. 1), 4-bit quantized | Local / HuggingFace |
| Llama-3.1-8B-Instant | Summarization (Ext. 2) | Groq API |

---

## Authors

Hasti Azadnia · Francesco Vanella · Ayda Ghasemazar · Sara Asadi Khomami  
*MSc Data Science & Engineering — Politecnico di Torino*

---

## Credits

This project builds on the **SigExt** model. See the [original repository](https://github.com/amazon-science/SigExt) for details.
