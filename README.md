# 🧠 AlgoBound: Evaluating OOD Generalization Bottlenecks in Neural Algorithmic Reasoning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3c8b3d)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Optimization-2c3e50)

Welcome to **AlgoBound**! This repository is a machine learning evaluation framework designed to answer a fundamental question: *Do neural networks actually learn algorithms, or are they just memorizing data?*

## 📖 Table of Contents
1. [The Problem: Algorithmic Alignment](#-the-problem-algorithmic-alignment)
2. [What AlgoBound Does](#-what-algobound-does)
3. [Supported Neural Architectures](#-supported-neural-architectures)
4. [Key Features](#-key-features)
5. [Installation](#️-installation)
6. [How to Use the Framework](#-how-to-use-the-framework-cli-guide)
7. [Project Structure](#-project-structure)
8. [Scientific Discoveries](#-scientific-discoveries)

---

## 🎯 The Problem: Algorithmic Alignment
Modern deep learning architectures (Transformers, GNNs) are incredible at interpolating data within their training distribution. However, when you ask them to extrapolate an algorithmic rule to a larger, Out-of-Distribution (OOD) size, they often collapse. 

For example: If a model learns how to find the shortest path on a 10-node graph, can it flawlessly apply that exact same logic to a 120-node graph? Usually, the answer is no. This framework exposes the hard mathematical and architectural bottlenecks causing these failures.

---

## ⚙️ What AlgoBound Does
AlgoBound acts as a strict, fault-tolerant testing arena. It separates evaluation into two isolated phases to prevent "cheating" (data leakage):
1. **Phase 1 (The Training Room):** The model is trained on procedurally generated, small-scale algorithmic tasks (e.g., sequences of length 10, or 20-node graphs). It uses Early Stopping to save its most optimal brain state.
2. **Phase 2 (The Final Exam):** The model's weights are frozen, and it is forced to run inference on massive, unseen OOD data (e.g., sequences up to length 500, or 120-node graphs). We measure the degradation in Accuracy or Mean Absolute Error (MAE).

---

## 🏛️ Supported Neural Architectures
The framework currently tests and benchmarks the following distinct neural architectures to isolate their specific mathematical inductive biases:

### 1. Sequence Models (Transformers & LSTMs)
Tested on **Cumulative XOR** (Boolean State-Tracking) and **Kadane's Algorithm** (Continuous Accumulation).
* **Standard Transformers (Absolute PE):** Tests the limits of absolute positional embeddings on unseen OOD sequence lengths.
* **Transformers with RoPE (Rotary Positional Embeddings):** Tests if relative distance calculations can solve the Transformer's inability to maintain a sequential running tally.
* **Long Short-Term Memory (LSTM):** Acts as the recurrent baseline. Flawless at boolean state-tracking, but exposes the "Continuous Memory Leak" when accumulating floating-point values over long time horizons.
* **State Space Models (Mamba - *In Development*):** Bridges the gap between Transformer parallelization and RNN sequential state-tracking.

### 2. Graph Routing Models (GNNs)
Tested on **Breadth-First Search (BFS)** and **Dijkstra's Algorithm** for spatial reasoning.
* **Shallow Message Passing Networks (Baseline GNN):** Standard 3-layer networks that expose the $k$-hop "Receptive Field Trap" (the physical inability to route information across graphs larger than the layer count).
* **Deep GNNs (The Champion Models):** Optuna-optimized 12-layer architectures designed to survive over-smoothing while increasing message-passing depth.
* **GNNs + Virtual Master Nodes:** A topological intervention that injects a global "ghost node" connected to all other nodes in the graph. This artificially reduces the maximum path distance between any two nodes to exactly 2 hops, mathematically shattering the $k$-hop bottleneck.

---

## ✨ Key Features
* **Unified CLI Switchboard:** A single `main.py` entry point controls all tasks, models, and data generation.
* **Dynamic Arena Mode:** Spin up custom "mutant" architectures via the command line and automatically race them against optimized Champion models.
* **Optuna Integration:** Automated Bayesian hyperparameter sweeps (`sweep.py`) to discover theoretically optimal learning rates and architectures.
* **Virtual Node Injection:** A toggleable graph modification to test global routing limits.

---

## 🛠️ Installation

Clone the repository and install the required dependencies.  
(We recommend using a virtual environment like `conda` or `venv`.)

```bash
git clone https://github.com/ElectroBoy0/AlgoBound.git
cd AlgoBound
pip install -r requirements.txt
💻 How to Use the Framework (CLI Guide)

AlgoBound uses a centralized Command Line Interface (main.py).
You do not need to modify Python files to run experiments — everything is controlled from the terminal.

🎛️ Core CLI Arguments

Before diving into the modes, here are the master parameters you can tune.

Core Routing

--task : algorithm to test (xor, kadane, bfs, dijkstra)

--model : architecture to run (transformer, lstm, gnn, rope, all)

Data & Scaling

--train_size : training dimension (e.g. 10 for sequence length or graph nodes)

--eval_sizes : OOD test sizes (example: 20 50 100 200)

Training Dynamics

--epochs : maximum training epochs (default 500)

--patience : early stopping trigger (default 15)

--lr : learning rate (default 0.00112)

Mode 1 — Publication Benchmark (Baseline vs Champion)

This mode generates publication-style ablation experiments.

Passing --model all will train:

a baseline architecture

the optimized champion model

Both are evaluated on identical OOD splits.

Example 1 — Testing Graph Routing (Dijkstra)
python main.py --task dijkstra --model all --train_size 20 --eval_sizes 20 40 80 120

Output

trains a 3-layer baseline GNN

trains the 12-layer champion model

produces a degradation curve saved to:

plots/dijkstra_comparison_curve.pdf
Example 2 — Testing Sequence State Tracking (XOR)
python main.py --task xor --model all --train_size 10 --eval_sizes 20 50 100

Output

Evaluates:

Absolute Transformers

RoPE Transformers

LSTMs

Demonstrates the parallel attention collapse phenomenon.

Mode 2 — Arena Mode (Custom Architecture Testing)

Arena Mode lets you test custom architectures directly from the CLI and race them against the champion model.

Additional Arena Flags

--num_layers : network depth

--hidden_dim : hidden dimension size

--use_vn : add a Virtual Node to GNN

--compare_champion : race your model against the champion

Example 3 — Racing a Custom GNN

Hypothesis:
A 5-layer GNN with 32 hidden dimensions can solve Dijkstra on 120 node graphs.

python main.py \
  --task dijkstra \
  --model gnn \
  --num_layers 5 \
  --hidden_dim 32 \
  --use_vn \
  --compare_champion \
  --lr 0.05 \
  --train_size 20 \
  --eval_sizes 20 40 80 120 \
  --epochs 300
🛡️ Champion Shield

AlgoBound automatically protects the champion configuration.

Your custom model may run at lr = 0.05, but the champion will always train at its optimal learning rate (0.00112) to ensure a fair comparison.

Mode 3 — Bayesian Hyperparameter Optimization

AlgoBound includes Optuna-based architecture search (sweep.py).

The optimizer searches across:

depth

hidden dimension

learning rate

to minimize OOD Mean Absolute Error.

Example 4 — Running a 30 Trial Sweep
python sweep.py --task dijkstra --trials 30

Output

Runs 30 training cycles and prints the optimal hyperparameter configuration.

📂 Project Structure
AlgoBound/
├── main.py          # CLI entrypoint and experiment router
├── models.py        # Neural architectures (GNN, Transformer, LSTM)
├── datasets.py      # Procedural dataset generators
├── trainer.py       # Training loops and evaluation
├── sweep.py         # Optuna hyperparameter search
├── utils.py         # Plotting and reproducibility utilities
├── requirements.txt # Python dependencies
├── plots/           # Generated experiment plots
└── results/         # Raw experiment CSV outputs
🔬 Scientific Discoveries
Track A — Sequences (XOR & Kadane)
Attention Collapse

Standard Transformers fail to track parity over long sequences due to the parallel attention mechanism.

LSTMs maintain 100% OOD accuracy.

Continuous Memory Leak

LSTMs perform well on discrete logic but degrade when accumulating floating-point values over long horizons (Kadane’s algorithm).

Track B — Graph Algorithms (BFS & Dijkstra)
Receptive Field Trap

A k-layer GNN can only propagate information k hops.

Thus a 3-layer GNN fails on large graphs.

Virtual Node Solution

Using:

deep GNNs (12 layers)

Virtual Master Node

AlgoBound significantly improves performance on 120-node graphs by enabling long-range communication.