# LLM Hands-on Introduction Course

**Mathematical Institute, University of Oxford**  [[Course Website](https://simonvary.github.io/llm-intro-course/)]

A short, four-week, code-first introduction to how modern decoder-only Large Language Models (LLMs) work end-to-end. Designed for DPhil students, this course takes an implementation-first approach. Students will build a minimal GPT step-by-step and experiment with open-weight models to understand architecture design choices and post-training objectives using PyTorch.

## Learning Outcomes
- End-to-end understanding of decoder LLM training and inference.
- Familiarity with architectural design choices in open models.
- Understand inference and long-context bottlenecks in practice.
- Run a minimal post-training pipeline (SFT + preferences).

## 📅 Schedule & Materials

| Lecture | Date (2026) | Topic | Materials |
| :--- | :--- | :--- | :--- |
| **1** | Mar 4 | **simpleGPT & basics**: History, tokenizer, tensors, causal self-attention, training, metrics | [Slides](./lecture1/slides/lecture1.pdf) • [Code](./lecture1/code) |
| **2** | Mar 11 | **Architecture design choices**: Position encoding (RoPE), attention variants (MQA/GQA), normalization | *Coming soon* |
| **3** | Mar 18 | **Inference & long context**: KV-cache, prefill vs. decode, long-context bottlenecks | *Coming soon* |
| **4** | Mar 25 | **Post-training**: Objectives, PEFT (LoRA), preference learning (DPO), RLVR, CoT | *Coming soon* |

## Prerequisites
Please bring a laptop with **Python** and **PyTorch** installed for the guided coding exercises.

## Instructor
**Simon Vary** 📧 [simon.vary@stats.ox.ac.uk](mailto:simon.vary@stats.ox.ac.uk) | 🌍