# Factual Inconsistency in Chart Captioning: Error Analysis and Two-Stage Fine-Tuning for MLLMs

This repository contains the code, model pipelines, and experimental resources for our paper:

**Factual Inconsistency in Chart Captioning: Error Analysis and Two-Stage Fine-Tuning for MLLMs**

## Overview

Chart captioning models often generate fluent but factually inconsistent descriptions.  
This project studies factual inconsistency in chart captioning from two aspects:

1. **Error Analysis**: We analyze factual error patterns across different model categories, including general MLLMs, chart MLLMs, and chart-specific models.
2. **Two-Stage Fine-Tuning**: We propose a two-stage framework that first trains models on factual consistency detection and then transfers this capability to chart caption generation.

Our work focuses on six factual error types:

- Value Error
- Label Error
- Trend Error
- Magnitude Error
- Out-of-Context (OOC) Error
- Nonsense Error

## Repository Structure

```text

├── dataset/                      # Dataset files and preprocessing resources
├── model/
│   ├── ChartInstruct-LLama2/     # Scripts/results for ChartInstruct-LLaMA2
│   ├── ChartVLM/                 # Scripts/results for ChartVLM
│   ├── Internlm-xcomposer2-vl-7b/# Scripts/results for InternLM-XComposer2-VL-7B
│   ├── LLaVA-7B/                 # Scripts/results for LLaVA
│   ├── MMCA/                     # Scripts/results for MMCA
│   ├── Qwen_VL_chat/             # Scripts/results for Qwen-VL / Qwen2.5-VL related experiments
│   ├── TinyChart/                # Scripts/results for TinyChart
│   ├── Unichart/                 # Scripts/results for UniChart
│   └── chartInstruction/         # Other chart instruction-based experiments
└── README.md
```

## Datasets

Experiments are conducted on the following datasets:

- **ChartX** – Used for error analysis and caption generation experiments.
- **ChartSumm** – Used for evaluating caption generation quality.
- **CHOCOLATE** – Used for factual consistency evaluation with ChartVE.

Due to licensing restrictions, some datasets are not redistributed in this repository.
Please download them from the original sources.


## Evaluation

We evaluate models using multiple metrics:

- **BERTScore F1**
- **BLEU**
- **ChartVE (Sentence-level entailment rate)**
- **Kendall's Tau** for caption ranking

These metrics evaluate both linguistic quality and factual consistency of generated chart captions.
