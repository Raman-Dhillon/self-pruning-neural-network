# 🧠 Self-Pruning Neural Network (PyTorch)

This project implements a neural network that **learns to prune itself during training** using learnable gating parameters.

---

## 🚀 Key Idea

Each weight is associated with a learnable gate:

w_eff = w × sigmoid(τ · g)

- g → learnable gate score  
- τ → temperature scaling (used to sharpen pruning)  
- Gate ≈ 0 → weight removed  
- Gate ≈ 1 → weight kept  

---

## 🧮 Loss Function

L_total = L_CE + λ · Sparsity Loss

- CrossEntropy → classification
- Sparsity Loss → encourages pruning

---

## ⚙️ Methodology

- Custom `PrunableLinear` layer (no use of torch.nn.Linear)
- Gates applied during forward pass
- L1-style regularization on gates
- Lambda scheduling for stable pruning

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.05   | 54.80       | 39.85         |
| 0.10   | 55.69       | 41.72         |
| 0.20   | 55.07       | 45.47         |

---

## 📈 Sparsity vs Accuracy

<p align="center">
  <img src="sparsity vs accuracy.png" width="500"/>
</p>

## 📉 Gate Distribution

<p align="center">
  <img src="gate value distrubution.png" width="500"/>
</p>

---

## 🔍 Observations

- Increasing λ increases sparsity
- High sparsity reduces accuracy
- Clear trade-off between efficiency and performance
- Temperature scaling enables near-binary gating

---

## 💡 Key Insight

Standard sigmoid produces smooth gating, which limits pruning.  
Applying temperature scaling sharpens decisions and enables effective sparsity.

---

## 🛠 Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

---

## 🚀 Future Work

- Extend to CNN architectures
- Use L0 regularization (Hard Concrete gates)
- Apply to real-world edge deployment
