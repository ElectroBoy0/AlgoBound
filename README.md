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

## What AlgoBound Does
AlgoBound acts as a strict, fault-tolerant testing arena. It separates evaluation into two isolated phases to prevent "cheating" (data leakage):
1. **Phase 1 (The Training Room):** The model is trained on procedurally generated, small-scale algorithmic tasks (e.g., sequences of length 10, or 20-node graphs). It uses Early Stopping to save its most optimal brain state.
2. **Phase 2 (The Final Exam):** The model's weights are frozen, and it is forced to run inference on massive, unseen OOD data (e.g., sequences up to length 500, or 120-node graphs). We measure the degradation in Accuracy or Mean Absolute Error (MAE).

---

## Supported Neural Architectures
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

## Key Features
* **Unified CLI Switchboard:** A single `main.py` entry point controls all tasks, models, and data generation.
* **Dynamic Arena Mode:** Spin up custom "mutant" architectures via the command line and automatically race them against optimized Champion models.
* **Optuna Integration:** Automated Bayesian hyperparameter sweeps (`sweep.py`) to discover theoretically optimal learning rates and architectures.
* **Virtual Node Injection:** A toggleable graph modification to test global routing limits.

---

## Installation

Clone the repository and install the required dependencies. (We recommend using a virtual environment like `conda` or `venv`).

```bash
git clone https://github.com/ElectroBoy0/AlgoBound.git
cd AlgoBound
pip install -r requirements.txt
```

***How to Use the Framework (CLI Guide)***
AlgoBound uses a highly dynamic, centralized Command Line Interface (main.py). You do not need to hardcode configurations or modify Python files to run experiments; you control the entire testing arena directly from your terminal.

Core CLI Arguments
Before diving into the modes, here are the master dials you can tune:

Core Routing:

`--task`: The algorithm to test (xor, kadane, bfs, dijkstra).

`--model`: The architecture to run (transformer, lstm, gnn, rope, or all).

Data & Scaling:

`--train_size`: The dimension to train on (e.g., 10 for sequence length 10 or 10-node graphs).

`--eval_sizes`: A list of massive OOD dimensions to test on (e.g., 20 50 100 200).

Training Dynamics:

`--epochs`: Maximum training epochs (Default: 500).

`--patience`: Early stopping trigger (Default: 15 epochs without improvement).

`--lr`: Learning rate (Default: 0.00112).

***Mode 1: The Publication Benchmark (Baseline vs. Champion)***

This mode is designed to generate publication-ready ablation studies. By passing --model all, the framework will sequentially train a standard baseline architecture and our scientifically optimized "Champion" architecture. It will then test them both on the exact same OOD data splits.

Example 1: Testing Graph Routing (Dijkstra's Algorithm)

```bash
python main.py --task dijkstra --model all --train_size 20 --eval_sizes 20 40 80 120
```
Output: The framework will train a shallow 3-layer GNN and our 12-layer dry-cherry-25 champion. It will automatically generate a comparative degradation curve and save it to plots/dijkstra_comparison_curve.pdf.

Example 2: Testing Sequence State-Tracking (Cumulative XOR)

```bash
python main.py --task xor --model all --train_size 10 --eval_sizes 20 50 100
```
Output: Evaluates Absolute Transformers, RoPE Transformers, and LSTMs, proving the parallel attention collapse.

***Mode 2: The Arena Mode (Custom Architecture Testing)***

Want to test your own architectural theory? You can spin up a custom "mutant" model directly in the terminal and physically race it against the reigning champion.

Additional Arena Flags:

`--num_layers`: Depth of your custom network.

`--hidden_dim`: Width of your custom network.

`--use_vn`: Injects a Virtual Master Node into your GNN.

`--compare_champion`: Automatically spawns the optimized champion model to race against your custom build.

Example 3: Racing a Custom GNN
Let's say you hypothesize that a 5-layer GNN with 32 dimensions is enough to solve Dijkstra on 120 nodes. You want to test it with a massive 0.05 learning rate:

```bash
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
  ```
The Champion Shield: AlgoBound's framework is fault-tolerant. In the command above, your custom model will train at lr=0.05 as requested. However, the framework will automatically "shield" the champion model and train it at its scientifically proven optimal rate (0.00112) to ensure a rigorously fair comparison.

***Mode 3: Bayesian Hyperparameter Optimization***

AlgoBound includes a dedicated Optuna script (sweep.py) to systematically discover new champion architectures. It searches across layer depth, hidden dimensions, and learning rates to minimize OOD Mean Absolute Error dynamically.

Example 4: Running a 30-Trial Sweep

```bash
python sweep.py --task dijkstra --trials 30
```
Output: The framework will run 30 isolated training/evaluation lifecycles, dropping the mathematically optimal hyperparameter dictionary directly into your terminal upon completion.

***Project Structure***
```plaintext
AlgoBound/
├── main.py              # The core CLI switchboard and experiment router
├── models.py            # Neural architectures (GNNs, Transformers, LSTMs)
├── datasets.py          # Procedural OOD data generation engines
├── trainer.py           # PyTorch training/evaluation loops with Early Stopping
├── sweep.py             # Optuna hyperparameter optimization script
├── utils.py             # Helper functions for reproducibility and PDF plotting
├── requirements.txt     # Python dependencies
├── plots/               # Auto-generated visual artifacts go here
└── results/             # Auto-generated raw CSV data goes here
```
***Scientific Discoveries***

***Track A: Sequences (XOR & Kadane's)***

***The Attention Collapse***: Standard Transformers (even with relative RoPE encodings) fail to track boolean parity over long sequences due to the parallel nature of attention. LSTMs maintain perfect 100% accuracy OOD.

***The Continuous Memory Leak***: While LSTMs dominate boolean logic, they suffer massive degradation when accumulating floating-point values over long time horizons (Kadane's algorithm).

***Track B: Graphs (BFS & Dijkstra's)***

***The Receptive Field Trap***: Standard 3-layer GNNs fail on massive graphs because information physically cannot travel further than 3 edges.

***The Solution***: By combining a 12-layer deep GNN with a Virtual Node (an artificial hub connected to all nodes), AlgoBound successfully bridges distant nodes and drastically reduces OOD degradation on massive 120-node graphs.