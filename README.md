# Fine-Tuning Large Language Models (LLMs)

## Overview
This project implements the fine-tuning of various **Large Language Models (LLMs)** for different NLP tasks, leveraging **Hugging Face Transformers** and efficient training techniques. The models are fine-tuned using **low-rank adaptation (LoRA), quantization**, and **accelerated training** to optimize resource usage.

### Tasks Covered:
- **Text Classification**: Fine-tuning **DistilBERT** and **Unsloth** for classification tasks.
- **Spam Detection**: Training DistilBERT to classify spam vs. ham emails.
- **Conditional Text Generation**: Fine-tuning **LLaMA** for generating context-aware text.

## Notebooks
### 1️⃣ Fine-Tuning DistilBERT for Text Classification
📌 **Notebook:** `fine_tuning_distilbert_for_products_classification.ipynb`
- Uses **DistilBERT**, a lightweight version of BERT, for binary classification.
- Prepares datasets and tokenizes text for training.
- Implements **quantization** and **optimized training techniques** to reduce memory consumption.
- Evaluates model accuracy and inference performance using standard metrics like accuracy.

### 2️⃣ Fine-Tuning DistilBERT for Spam Classification
📌 **Notebook:** `fine_tuning_distilbert_for_spam_mails_classification.ipynb`
- Fine-tunes **DistilBERT** on a real-world spam email detection dataset.
- Implements **preprocessing, tokenization, and dataset transformation** for effective training.
- Uses **Hugging Face Trainer API** for streamlined model fine-tuning and evaluation.
- Saves the fine-tuned model, enabling efficient inference on new email samples.

### 3️⃣ Fine-Tuning LLaMA for Conditional Text Generation
📌 **Notebook:** `fine_tuning_llama_for_conditional_text_generation.ipynb`
- Fine-tunes **LLaMA**, a powerful transformer model, for context-aware text generation.
- Uses **LoRA-based fine-tuning** to optimize memory usage and allow efficient adaptation.
- Implements **chat-like prompt formatting** to train the model for interactive and structured responses.
- Generates high-quality, meaningful responses based on contextual input.

### 4️⃣ Fine-Tuning LLM for Text Generation using Unsloth
📌 **Notebook:** `fine_tuning_llm_for_text_generation_ unsloth.ipynb`
- Utilizes **Unsloth**, an optimized version of LLaMA, for **high-speed fine-tuning**.
- Performs **dataset preparation, tokenization, and training** using efficient memory management techniques.
- Optimizes model efficiency with **quantization, mixed-precision (FP16) computation, and low-rank adaptation (LoRA)**.
- Enables faster fine-tuning while maintaining high accuracy, reducing computational overhead.

## Model Performance & Optimization
- Uses **LoRA** (Low-Rank Adaptation) to make fine-tuning lightweight and efficient.
- **Quantization** reduces model size while retaining performance, enabling deployment on smaller hardware.
- **Trainer API** from Hugging Face provides structured fine-tuning with logging and evaluation.
- **FP16 precision & BitsAndBytes optimization** speed up training while minimizing memory footprint.

## Conclusion
This project provides practical insights into **fine-tuning LLMs for real-world applications**, optimizing for **memory efficiency and computational performance** for many tasks such as: **classification, spam detection, or text generation**.