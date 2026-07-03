# Communication-Efficient Decentralized Federated Learning

Multimodal Information Processing Systems (MIPS) Lab | Mentor: Prof. Rajesh Hegde, Department of Electrical Engineering

## Overview

This project implements a decentralized federated learning (DFL) framework designed to reduce communication overhead among clients while preserving model accuracy under non-IID data distributions. The framework simulates a 40-node decentralized network training a CNN on CIFAR-10 (with an additional MNIST experiment), using sparse communication instead of dense weight synchronization at every round.

## Method

- **Data partitioning:** Dirichlet-based non-IID partitioning of CIFAR-10 across 40 clients (`alpha = 0.3`)
- **Local training:** FedProx-style local training with an elastic anchor (proximal) term to stabilize updates under sparsity
- **Sparse communication:** Layer-wise sparse yielding of model deltas (mechanical/thermal yield ratios), instead of transmitting full dense updates
- **Residual memory:** Residual/strain memory accumulated on local deltas (not absolute weights) to recover information dropped during sparsification
- **Aggregation:** Mask-aware aggregation that accounts for which parameters were actually transmitted by each client
- **Topology:** Decentralized peer-to-peer graph (Watts-Strogatz style, `k=2`, `p=0.1`), with an optional adaptive topology mode (shortcut edges, stress-based rewiring)
- **Baseline:** Benchmarked against a dense decentralized baseline over 100 communication rounds

## Results

- Achieved **75% lower communication overhead** while maintaining accuracy within **3%** of the dense decentralized baseline
- Demonstrated that sparse update transmission preserves decentralized training scalability with minimal loss in model performance

## Tools Used

- PyTorch, Torchvision
- NetworkX (peer-to-peer topology construction)
- NumPy, Matplotlib
- scikit-learn (MNIST prototype)
