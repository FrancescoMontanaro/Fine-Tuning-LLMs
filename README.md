# Fine-Tuning Large Language Models

## Overview
This repository collects hands-on experiments for adapting open-source large language models (LLMs) to downstream tasks. Each notebook walks through data preparation, training, evaluation, and inference while demonstrating different alignment strategies such as supervised fine-tuning (SFT), direct preference optimization (DPO), and online reinforcement learning (ORL). Hugging Face Transformers, PEFT/LoRA adapters, and lightweight generation utilities power the workflows.

The notebooks emphasize reproducible training scripts, modular helpers under `src/`, and lightweight storage of datasets and artifacts so experiments remain easy to rerun or extend.

## Repository Layout

```text
.
├── datasets/                         # Local copies of datasets used by the notebooks
├── notebooks/
│   ├── supervised_fine_tuning/
│   │   ├── sft_instruction_following.ipynb
│   │   ├── sft_conditional_text_generation.ipynb
│   │   ├── sft_emojis_translation.ipynb
│   │   ├── sft_text_generation_unsloth.ipynb
│   │   ├── sft_products_classification.ipynb
│   │   └── sft_spam_mails_classification.ipynb
│   ├── direct_preference_optimization/
│   │   └── dpo_preference_alignment.ipynb
│   └── online_reinforcement_learning/
│       ├── orl_grpo_alignment.ipynb
│       └── trainer_output/           # Run logs and checkpoints from GRPO experiments
├── requirements.txt                  # Shared Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Chat template builders and generation utilities
│   ├── hf.py                         # Hugging Face model/tokenizer/dataset helpers
│   └── utils.py                      # Miscellaneous helpers reused across notebooks
└── venv/                             # Optional local virtual environment (ignored by git)
```

## Notebook Catalogue

| Technique | Notebook | Key Topics |
|-----------|----------|------------|
| SFT | `sft_instruction_following.ipynb` | Instruction-following SFT with chat templates, LoRA adapters, and evaluation on dialogue-style prompts. |
| SFT | `sft_conditional_text_generation.ipynb` | Conditional generation over scientific abstracts with prompt formatting, validation sampling, and optional adapter export. |
| SFT | `sft_emojis_translation.ipynb` | Emoji ↔︎ natural language translation on a custom dataset, highlighting data curation and prompt engineering. |
| SFT | `sft_text_generation_unsloth.ipynb` | Memory-efficient SFT using Unsloth, 4-bit quantization, gradient checkpointing, and fast HF integration. |
| SFT | `sft_products_classification.ipynb` | Multi-class product title classification with scikit-learn metrics, confusion matrices, and dataset preprocessing. |
| SFT | `sft_spam_mails_classification.ipynb` | Spam vs ham email detection using the Trainer API, quantization-ready setup, and evaluation utilities. |
| DPO | `dpo_preference_alignment.ipynb` | Direct Preference Optimization workflow spanning dataset prep, reward modeling, and preference-aligned fine-tuning. |
| ORL | `orl_grpo_alignment.ipynb` | Online reinforcement learning with GRPO (Grouped Reward Policy Optimization). |

## Shared Utilities

- `src/data_processing.py` centralizes chat template construction, supervised signal masking, and generation helpers used across multiple notebooks.
- `src/hf.py` offers wrappers for loading models, tokenizers, and datasets with consistent configuration and sensible defaults.
- `src/utils.py` contains reusable utilities (seeding, logging, filesystem helpers) to keep notebooks focused on experimentation.

Import the package either by adjusting `sys.path` inside notebooks or by installing it in editable mode:

```bash
pip install -e .
```

## Datasets & Artifacts

- Place local or downloaded datasets under `datasets/`. Several notebooks expect CSV files such as `arxiv_dataset.csv` or `emoji_translation_dataset.csv` to reside here.
- Training outputs (checkpoints, logs) from reinforcement learning runs currently live in `notebooks/online_reinforcement_learning/trainer_output/`. Consider moving long-lived artifacts under a dedicated `artifacts/` directory if storage grows.
- Saved adapters or fine-tuned weights can be written to a `saved_models/` directory (not tracked by default) to keep the repository lightweight.

## Environment Setup

1. Create a virtual environment (Python 3.10+ recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the Notebooks

- Launch Jupyter Lab/Notebook or VS Code, select the `fine-tuning-llms` kernel, and open the notebook of interest.
- Follow the dataset preparation steps documented in each notebook; many cells load from `datasets/` or download sources via Hugging Face Hub.
- When training, monitor GPU memory usage and adjust LoRA ranks, batch sizes, or quantization settings as demonstrated in the notebook templates.