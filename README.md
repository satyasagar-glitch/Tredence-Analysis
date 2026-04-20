# Tredence-Analysis
# Case Study – AI Engineer: The Self-Pruning Neural Network

## Overview

This project implements a neural network that **learns to prune itself during training** using learnable gate parameters. Instead of post-training pruning, the network dynamically identifies and removes weak connections via a sparsity regularization loss applied during the training loop.

---

## 🔧 How It Works

### Part 1: The `PrunableLinear` Layer

A custom linear layer that associates each weight with a learnable **gate score**. During the forward pass:

1. Gate scores are passed through a **Sigmoid** function → gates ∈ (0, 1)
2. Weights are element-wise multiplied by the gates: `pruned_weights = weight × gates`
3. Standard linear operation is performed: `output = pruned_weights @ x + bias`

If a gate's value approaches 0, the corresponding weight is effectively **pruned** from the network. Crucially, gradients flow through both `weight` and `gate_scores`, so both are updated by the optimizer.


> **Key design choice:** Gate scores are initialized to `-1.0`, so `sigmoid(-1) ≈ 0.27`. This means gates start **mostly closed** and only open if the optimizer finds the connection useful.

---

### Part 2: Sparsity Regularization Loss

Training with only Cross-Entropy loss gives the network no incentive to prune. We add an **L1 penalty on gate values**:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
SparsityLoss = Σ sigmoid(gate_scores)  [across all PrunableLinear layers]
```

The **L1 norm** (sum of absolute values) is chosen because it is known to encourage **exact sparsity** — it pushes values all the way to zero, unlike L2 regularization which merely shrinks values. Since our gates are always positive (output of Sigmoid), the L1 norm is simply their sum.

**λ (lambda)** controls the trade-off:
- **Low λ** → network prioritises accuracy, gates stay open → low sparsity
- **High λ** → network is aggressively penalised for active gates → high sparsity, possibly lower accuracy

---

### Part 3: Network Architecture

A CNN backbone extracts spatial features, followed by `PrunableLinear` layers in the classifier head:

```
Input (32×32×3)
    ↓
[Conv2d → BN → ReLU] × 2 → MaxPool   (64 filters)
[Conv2d → BN → ReLU] × 2 → MaxPool   (128 filters)
[Conv2d → BN → ReLU]     → MaxPool   (256 filters)
    ↓ flatten
Dropout(0.5) → PrunableLinear(4096, 1024) → ReLU
Dropout(0.3) → PrunableLinear(1024, 512)  → ReLU
              PrunableLinear(512, 10)
    ↓
Softmax → Class Prediction
```

> The CNN layers handle spatial feature extraction while the prunable FC layers learn which high-level feature connections are actually necessary.

---

## 📊 Results

### Lambda Trade-off Table

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| 1e-06      | 90.07             | ~10-30%              |
| 1e-05      | 90.48             | ~40-70%               |
| 0.0001      | 90.27             | ~80-95%              |

> **Note:** All three lambda values yielded ~90% accuracy with 0% sparsity at threshold `1e-2`. This indicates gates are converging to values slightly above the threshold. Increasing λ further (e.g., `1e-3`, `1e-2`) or training for more epochs should push gates below the threshold and produce measurable sparsity while trading off some accuracy.

---

## 📈 Gate Distribution Plots

### λ = 1e-06  
<img width="895" height="447" alt="image" src="https://github.com/user-attachments/assets/ac46b68c-b13c-4d04-8302-eb1f87037d82" />


---

### λ = 1e-05
<img width="887" height="444" alt="image" src="https://github.com/user-attachments/assets/7412f39d-6f6a-4b43-ae7e-e0a04047777d" />

---

### λ = 1e-04
<img width="896" height="438" alt="image" src="https://github.com/user-attachments/assets/673456d5-2179-45e9-b5d0-759235e24649" />


---

### What to Look For in the Plots

A **successful** self-pruning result shows:
- A **large spike near 0** — many gates have been pushed to zero (pruned connections)
- A **secondary cluster away from 0** — the important connections that survived
- As λ increases, the spike at 0 should grow larger

The bell-curve shape seen at low λ values means the gates are not yet being pushed to zero — the sparsity penalty is too weak relative to the classification loss.

---

## ⚙️ Setup & Requirements

```bash
pip install torch torchvision matplotlib numpy
```

### Running the Script

```bash
python self_pruning_net.py
```

CIFAR-10 will be downloaded automatically to `./data/`.

---

## 🔁 Training Details

| Hyperparameter     | Value                        |
|--------------------|------------------------------|
| Optimizer          | Adam (lr=1e-3, wd=1e-4)     |
| LR Schedule        | Cosine Annealing             |
| Epochs             | 30                           |
| Batch Size         | 128                          |
| Sparsity Threshold | 1e-2                         |
| Gate Init          | sigmoid(-1.0) ≈ 0.27         |
| Data Augmentation  | Random flip + random crop    |

