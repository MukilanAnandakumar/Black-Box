# Black-Box Query-Based Adversarial Attack on MNIST

This project provides a research-grade framework for evaluating the robustness of Deep Learning models against black-box adversarial attacks. It implements the **Natural Evolution Strategy (NES)** for gradient estimation in a query-limited setting and compares it against standard white-box attacks.

## Core Features

- **Modular Deep Learning**: Support for multiple CNN architectures.
- **Advanced Attack Suite**:
  - **Black-Box (NES)**: Untargeted and Targeted attacks.
  - **White-Box (FGSM)**: Baseline comparison.
- **Robustness & Defense**: Implementation of **Adversarial Training**.
- **Research Experiments**:
  - Transferability analysis between different architectures.
  - Confidence drop and distortion metrics.
  - Cumulative success rate vs. query budget analysis.
- **Publication-Ready Visualization**: Automated generation of plots and structured result logging (JSON).

## Project Structure

- `data_loader.py`: Efficient dataset loading and normalization.
- `model.py`: Architecture definitions (Model A and Model B).
- `train.py`: Training script with support for adversarial training.
- `attack.py`: Implementation of FGSM and NES attacks.
- `evaluate.py`: Main experimentation script.
- `utils.py`: Metrics, logging, and plotting utilities.
- `results/`: Directory for experimental outputs and figures.
- `PROJECT_EXPLANATION.txt`: Detailed technical breakdown of each component.
- `FLOW_DIAGRAM.md`: Visual Mermaid-based project workflow diagram.
- `RESEARCH_PAPER.txt`: Formal publication-style documentation.

## Prerequisites

- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib
- NumPy

## Quick Start

1. **Dataset Setup**:
   Ensure the MNIST dataset is located in `./MNIST/Training` and `./MNIST/Testing`.

2. **Train Models**:
   ```bash
   python train.py
   ```
   This will train the base victim (Model A), a deeper model (Model B), and a robust model (Adversarial Training).

3. **Run Experiments**:
   ```bash
   python evaluate.py
   ```
   This script performs the full research evaluation and saves results to the `results/` folder.

## Threat Model

The attack operates under a **Strict Black-Box** setting:
- **No access** to model weights, architecture, or gradients.
- **Query-only access**: The attacker only observes the Softmax output probabilities.
- **Query Budget**: Attacks are constrained by a maximum number of queries per sample.
- **Perturbation Constraint**: Adversarial examples are bounded by an $L_\infty$ norm (Epsilon).

## Real-World Relevance

Adversarial attacks pose significant risks to AI systems in cybersecurity, autonomous driving, and financial fraud detection. This project illustrates how even "hidden" models can be compromised through minimal interactions, emphasizing the need for robust defense mechanisms like adversarial training.

