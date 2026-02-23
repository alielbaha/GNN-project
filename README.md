# Curvature-Aware Graph Neural Networks

### Exploring the role of geometric curvature in graph representation learning

---

This repository contains three progressive experiments investigating how **graph curvature** ( specifically **Forman-Ricci curvature)** can improve **Graph Convolutional Networks (GCNs)** on  **heterophilic** datasets while perserving good results on **homophilic** datasets.

  

The central idea is that **curvature reveals local geometric structure** in graphs, which can be used to **guide message passing** in GCNs. By incorporating curvature either as fixed weights, learnable signals, or dynamic gates, these experiments demonstrate how GNNs can adapt more efficiently to graph topology.

---

## Experiments summary

### **Experiment 1: Comparing Standard GCN vs Curvature-Weighted GCN**

**File:** `experiment_1.ipynb`
**Goal:** Assess whether using **fixed curvature-based edge weights** improves node classification performance.

#### Models

* **Vanilla GCN:** Standard message-passing using adjacency only.
* **Curv-GCN:** Same as Vanilla GCN but with **Forman-Ricci curvature edge weights**.

#### Setup

* Datasets:

  * **Cora** (homophilic)
  * **Chameleon** (heterophilic)
* Hidden dimension: 64
* Optimizer: Adam (lr = 0.01, weight_decay = 5e-4)
* Metric: Test accuracy averaged over 5 runs

#### Outcome

* Curv-GCN performs comparably on **Cora** (smooth structure).
* Curv-GCN improves on **Chameleon** (mixed, noisy connectivity).

---

### **Experiment 2: Introducing Fixed vs Learnable Curvature**

**File:** `experiment_2.ipynb`
**Goal:** Determine whether learning curvature values dynamically enhances generalization across graphs.

#### Models

1. **Vanilla GCN:** Baseline without curvature.
2. **Fixed Curv-GCN:** Uses normalized, precomputed curvature values as fixed edge weights.
3. **Learnable Curv-GCN:** Initializes with curvature but allows **gradient-based updates** during training.

#### Setup

* Datasets: Cora and Chameleon
* Random 60/20/20 train/val/test split
* 5 independent runs for statistical robustness

#### Outcome

| Model              | Behavior                  | Benefit                                      |
| ------------------ | ------------------------- | -------------------------------------------- |
| Vanilla GCN        | Ignores curvature         | Baseline performance                         |
| Fixed Curv-GCN     | Adds geometric prior      | Better edge weighting on heterophilic graphs |
| Learnable Curv-GCN | Learns curvature dynamics | adaptability                                 |

---

### **Experiment 3: Curvature-Gated GCN (CG-GCN)**

**File:** `experiment_3.ipynb`
**Goal:** Develop a dual-path GCN that dynamically routes information through **homophilic** or **heterophilic** channels based on local curvature.

#### Model Architecture

* Two parallel convolutional paths:

  * **Homophilic Path:** Standard GCNConv layers for smooth feature propagation.
  * **Heterophilic Path:** Custom HeteroConv layers for contrasting information aggregation.
    

```mermaid
flowchart TD
    A[Input Features] --> B[Compute Forman Curvature<br/>edge]
    B --> C[Homophilic Path<br/>GCNConv]
    B --> D[Heterophilic Path<br/>HeteroConv]
    C --> E[Gated Combination]
    D --> E
    E --> F[Node Class Predictions]
    
* The two outputs are combined adaptively per edge.

#### Setup

* Datasets: Cora and Chameleon
* Optimizer: Adam, lr = 0.01
* Epochs: 200

#### Expected Outcome

* Curvature-Gated GCN adapts seamlessly to both homophilic (Cora) and heterophilic (Chameleon) structures.
* Outperforms curvature-fixed and vanilla models through **localized geometric adaptation**.

---

## Datasets Description

| Dataset       | Type         | Nodes | Edges  | Classes | Description                                 |
| ------------- | ------------ | ----- | ------ | ------- | ------------------------------------------- |
| **Cora**      | Homophilic   | 2,708 | 5,429  | 7       | Citation network of scientific papers       |
| **Chameleon** | Heterophilic | 2,277 | 36,101 | 5       | Wikipedia pages on chameleon-related topics |

> Both datasets are obtained from PyTorch Geometric.


## Project structure

```
curvature_gnn/
│
├── experiment_1.ipynb          # GCN vs Curv-GCN comparison
├── experiment_2.ipynb          # Fixed vs Learnable Curvature
├── experiment_3.ipynb          # Curvature-Gated GCN
│
├── models/
│   ├── gcn.py               # Vanilla GCN implementation
│   ├── curvature_gcn.py     # Curv-GCN + Learnable Curv-GCN
│   ├── curvature_gated.py   # Curvature-Gated GCN
│
├── README.md           
└── requirements.txt         
```

---

## Reproducibility

1. **Install dependencies:**

   ```bash
   pip install torch torch_geometric numpy matplotlib
   ```
2. **Run an experiment:**

   ```bash
   python experiment_1.py
   python experiment_2.py
   python experiment_3.py
   ```
