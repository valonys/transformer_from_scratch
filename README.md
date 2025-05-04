# Transformer from Scratch

A PyTorch implementation of a modern Transformer with sliding-window multi-head attention, Rotary Position Embeddings (RoPE), and Mixture-of-Experts (MoE). Trained on WikiText-2 for language modeling.

## Features
- Sliding-window attention: O(N·w) complexity, window size = 7.
- RoPE: Relative positional encoding for better generalization.
- MoE: Top-2 expert selection for efficient feed-forward layers.
- Trained on WikiText-2 (character-level) for 3 epochs.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
