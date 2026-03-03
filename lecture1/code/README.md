# Lecture 1 Code: Foundations of Large Language Models

This directory contains the practical coding exercises and implementations for Lecture 1.

## Jupyter Notebook Exercises

- **Text Processing & Tokenization:** Building a vocabulary from raw text, mapping strings to integer sequences (encoding/decoding), and structuring the data into context windows for autoregressive learning.
- **Attention Mechanisms:** Hands-on implementation of the multi-head self-attention. You will write the code to compute Queries, Keys, and Values, apply the scaled dot-product attention, and use lower-triangular masking to ensure the model only attends to past context.
- **Transformer Building Blocks:** Construction of the individual neural network modules: MHA layer, the Feed-Forward Network, and Layer Normalization.

## The `simpleGPT` Implementation

In the `simpleGPT/` directory is a minimalistic implementation of our GPT-style language model, bringing together all the concepts from the notebooks.


## How to Use

Start with the **Jupyter notebooks** to understand the tensor operations and architecture.

Once you are comfortable with the individual building blocks, transition to the **`simpleGPT` folder** to see how these components are integrated and trained in a continuous loop.

Download TinyStories:

```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true -O TinyStories-train.txt
```

Train with:

```bash
python simpleGPT/train.py
```