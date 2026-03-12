# LLM Hands-on Introduction Course

**Mathematical Institute, University of Oxford**  [[Course Website](https://simonvary.github.io/llm-intro-course/)]

A short, four-week, introduction to decoder-only Large Language Models (LLMs). Students will build a minimal GPT step-by-step and experiment with open-weight models to understand architecture design choices and post-training objectives using PyTorch.

## Learning Outcomes
- Understanding of decoder LLM training and inference.
- Familiarity with architectural design choices in open models.
- Understand inference and long-context bottlenecks in practice.
- Run a minimal post-training pipeline (SFT + preferences).

## 📅 Schedule & Materials

| Lecture | Date (2026) | Topic | Materials |
| :--- | :--- | :--- | :--- |
| **1** | Mar 4 | **simpleGPT & basics**: History, tokenizer, tensors, causal self-attention, training, metrics | [Slides](https://simonvary.github.io/llm-intro-course/lecture1/slides/lecture1_handout.pdf) • [Code](./lecture1/code) |
| **2** | Mar 11 | **Architecture design choices**: Recap of MHA + Transformer, Position encoding RoPE, normalization (pre/post-norm), dimensions | [Slides](https://simonvary.github.io/llm-intro-course/lecture1/slides/lecture2_handout.pdf) • [Code](./lecture2/code) |
| **3** | Mar 18 | **Inference & long context**: KV-cache, prefill vs. decode, long-context bottlenecks | |
| **4** | Mar 25 | **Post-training**: Objectives, PEFT (LoRA), preference learning (DPO), RLVR, CoT |  |

## Prerequisites
Please bring a laptop with **Python** and **PyTorch** installed for the guided coding exercises.

## Instructor
**Simon Vary** [simon.vary@stats.ox.ac.uk](mailto:simon.vary@stats.ox.ac.uk) | [simonvary.github.io](https://simonvary.github.io/)