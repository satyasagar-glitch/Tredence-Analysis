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

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -1.0))

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)
```

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
| 1e-06      | 90.07             | 0.00               |
| 1e-05      | 90.48             | 0.00               |
| 0.0001      | 90.27             | 0.00               |

> **Note:** All three lambda values yielded ~90% accuracy with 0% sparsity at threshold `1e-2`. This indicates gates are converging to values slightly above the threshold. Increasing λ further (e.g., `1e-3`, `1e-2`) or training for more epochs should push gates below the threshold and produce measurable sparsity while trading off some accuracy.

---

## 📈 Gate Distribution Plots

### λ = 1e-06  
<img width="896" height="487" alt="image" src="https://github.com/user-attachments/assets/8756ad37-ff9d-492d-b56c-93df6b60f016" />


---

### λ = 1e-05
<img width="897" height="485" alt="image" src="https://github.com/user-attachments/assets/4af8704a-51be-4476-b735-1c3c6d09ff7c" />

---

### λ = 1e-04
<img width="884" height="481" alt="image" src="https://github.com/user-attachments/assets/88d8e112-342e-4e70-8046-b17109a02f0b" />


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

---

## 💡 Why L1 on Sigmoid Gates Encourages Sparsity

The L1 penalty creates a **constant gradient** that pushes gate values toward zero regardless of their magnitude. Unlike L2 (which applies a weaker gradient as values approach zero), L1 applies the same force all the way down — making it capable of driving values to **exactly zero**.

Since our gates are always positive (Sigmoid output), minimizing their L1 sum directly minimizes the number of active connections. The optimizer faces a clear choice per gate: *"Is this connection useful enough to overcome the constant λ penalty?"* If not, the gate collapses to zero.

---
