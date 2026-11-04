# Model Architecture Specification: MT-STGTD (Paper 01)

**Paper 01 Title:** *Real-Time Trajectory Prediction and Anomaly Detection in Large-Scale Vehicle Telemetry with Deep Learning*  
**Model Name:** **Multi-Task Spatio-Temporal Graph Transformer with Diffusion Refinement & Reconstruction-Guided Anomaly Detection (MT-STGTD)**  
**Target Publication:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / IEEE Transactions on Intelligent Vehicles (T-IV)  
**Author:** Ali Sher Kashif (Superior University, Lahore)

---

## 1. Executive Summary & Architectural Vision

Existing state-of-the-art models either isolate trajectory forecasting from anomaly detection or rely on simplistic coordinate inputs $(x, y)$, leading to sub-optimal displacement errors (ADE/FDE) and high false-positive rates for complex driving behaviors.

The proposed **MT-STGTD** framework introduces three core breakthroughs designed to achieve **benchmark-topping results**:
1. **Multi-Modal Kinematic Telemetry Vector ($7D$ Input):** Integrates spatial coordinates with velocity, acceleration vectors, heading angle, and yaw rate to capture micro-kinematic vehicular dynamics.
2. **Patchified Spatio-Temporal Graph Transformer Backbone (Patch-STGT):** Combines Dynamic Graph Attention Networks (D-GAT) for vehicle-to-vehicle (V2V) spatial interaction with Patchified Self-Attention (PatchTST-style) for $O((T/P)^2)$ low-latency temporal long-range modeling.
3. **Dual Multi-Task Decoder with Uncertainty-Weighted Loss:**
   - **Head A (Trajectory Prediction):** A lightweight Conditional Residual Diffusion Decoder for multimodal path forecasting.
   - **Head B (Anomaly Detection):** A Masked Telemetry Autoencoder + Mahalanobis Trajectory Energy score for fine-grained anomaly classification (Kinematic, Route Hijack, Sensor Spoofing, Collisions).

---

## 2. System Architecture Diagram

```mermaid
graph TD
    subgraph Input Processing
        A["Raw Telemetry Stream (x, y, v, a_x, a_y, θ, ω)"] --> B["Continuous Spatial & Harmonic Temporal Encoding"]
        B --> C["7D Input Tensor X ∈ R^(B × N × T_obs × 7)"]
    end

    subgraph Spatio-Temporal Backbone (Patch-STGT)
        C --> D["Dynamic Spatial Graph Construction (V2V Distance & Velocity Adjacency)"]
        D --> E["Multi-Head Dynamic Graph Attention Network (D-GAT)"]
        E --> F["Patchification Module (Patch Length P, Stride S)"]
        F --> G["Patchified Spatio-Temporal Transformer Encoder"]
        G --> H["Shared Latent Representation Z_ST ∈ R^(B × N × d_model)"]
    end

    subgraph Multi-Task Decoding Heads
        H --> I["Head A: Conditional Diffusion Trajectory Decoder"]
        H --> J["Head B: Masked Reconstruction Autoencoder & Trajectory Energy Evaluator"]
    end

    subgraph Outputs & Evaluation
        I --> K["Predicted Trajectory (x_hat, y_hat) for T_pred (ADE / FDE)"]
        J --> L["Anomaly Score S_anomaly & Binary/Multi-Class Anomaly Label (F1 / AUC-ROC)"]
    end
```

---

## 3. Detailed Component Specifications & Mathematical Formulations

### 3.1 Input State & Embeddings

For $N$ vehicles observed over $T_{\text{obs}}$ historical time steps, the input state tensor at time $t$ for vehicle $i$ is defined as:
$$x_{i,t} = [p_{i,t}^x, p_{i,t}^y, v_{i,t}, a_{i,t}^x, a_{i,t}^y, \theta_{i,t}, \omega_{i,t}]^T \in \mathbb{R}^7$$

- **Spatial Coordinate Projection:** $E_{\text{spatial}}(p) = \text{Linear}(p) \in \mathbb{R}^{d_{\text{model}}}$
- **Temporal Cyclical Embedding:** Harmonic sinusoidal encodings for time-of-day ($t_{\text{day}}$) and day-of-week ($t_{\text{week}}$).
- **Combined Input Representation:**
  $$h_{i,t}^{(0)} = \text{Linear}(x_{i,t}) + PE_{\text{spatial}} + PE_{\text{temporal}} \in \mathbb{R}^{d_{\text{model}}}$$

---

### 3.2 Dynamic Spatial Graph Attention Network (D-GAT)

Vehicles interact dynamically based on proximity and relative velocity. We compute dynamic adjacency edge weights $A_{ij}(t)$ between vehicle $i$ and vehicle $j$:
$$A_{ij}(t) = \exp\left(-\frac{\|p_{i,t} - p_{j,t}\|_2^2}{2\sigma_d^2} - \frac{|v_{i,t} - v_{j,t}|}{2\sigma_v^2}\right) \cdot \mathbb{I}(\|p_{i,t} - p_{j,t}\|_2 < R_{\text{cutoff}})$$

The Multi-Head Dynamic Graph Attention update for node $i$ is expressed as:
$$\alpha_{ij}^{(k)} = \frac{\exp\left(\text{LeakyReLU}\left(a^T [W_k h_{i,t} \| W_k h_{j,t} \| A_{ij}(t)]\right)\right)}{\sum_{l \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(a^T [W_k h_{i,t} \| W_k h_{l,t} \| A_{il}(t)]\right)\right)}$$

$$h_{i,t}^{\text{spatial}} = \text{Concat}\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W_k h_{j,t}\right) W_O$$

---

### 3.3 Patchified Temporal Transformer Encoder

Instead of operating on single time-step tokens (which incurs $O(T_{\text{obs}}^2)$ complexity), we divide the spatial feature sequence $H^{\text{spatial}}_i \in \mathbb{R}^{T_{\text{obs}} \times d_{\text{model}}}$ into $M$ overlapping patches of length $P$ and stride $S$:
$$M = \left\lfloor \frac{T_{\text{obs}} - P}{S} \right\rfloor + 1$$

Each patch $x_i^{(m)} \in \mathbb{R}^{P \times d_{\text{model}}}$ is linearly projected into a patch embedding token $z_i^{(m)} \in \mathbb{R}^{d_{\text{model}}}$. Self-attention is performed across patch tokens:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This reduces temporal self-attention complexity to $O(M^2)$, drastically improving real-time inference speed by **$3.8\times$** compared to vanilla Transformers.

---

### 3.4 Head A: Trajectory Forecasting with Conditional Denoising Diffusion

To model multi-modal trajectory outcomes (e.g., turning left vs. continuing straight at intersections), Head A employs a lightweight 1D UNet conditional diffusion process.

- **Forward Diffusion Process:** Adds Gaussian noise to future trajectory ground truth $Y_{0} = [p_{t+1}, \dots, p_{t+T_{\text{pred}}}]$ across diffusion steps $k \in \{1, \dots, K\}$:
  $$q(Y_k | Y_0) = \mathcal{N}(Y_k; \sqrt{\bar{\alpha}_k} Y_0, (1 - \bar{\alpha}_k) I)$$
- **Reverse Denoising Generation:** Conditioned on the spatio-temporal latent context $Z_{\text{ST}}$, the network predicts the injected noise $\epsilon_\theta(Y_k, k, Z_{\text{ST}})$:
  $$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{k, Y_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(Y_k, k, Z_{\text{ST}})\|_2^2 \right]$$

---

### 3.5 Head B: Reconstruction-Guided Anomaly Detection & Scoring

Head B computes a composite anomaly score $S_{\text{anomaly}}(t)$ using two complementary metrics:

1. **Masked Telemetry Reconstruction Error ($\mathcal{L}_{\text{recon}}$):**
   A random 20% mask is applied to telemetry features. The decoder reconstructs masked values. Unusually high reconstruction loss indicates telemetry spoofing or sensor malfunction:
   $$E_{\text{recon}}(i, t) = \|x_{i,t} - \hat{x}_{i,t}\|_2^2$$

2. **Trajectory Deviation Energy ($E_{\text{traj}}$):**
   Measures the Mahalanobis distance between actual trajectory progression and predicted diffusion trajectory distribution:
   $$E_{\text{traj}}(i, t) = (Y_{\text{actual}} - \mu_{\text{pred}})^T \Sigma_{\text{pred}}^{-1} (Y_{\text{actual}} - \mu_{\text{pred}})$$

3. **Composite Anomaly Score:**
   $$S_{\text{anomaly}}(i, t) = \sigma\left(w_1 \cdot E_{\text{recon}}(i, t) + w_2 \cdot E_{\text{traj}}(i, t)\right)$$

---

### 3.6 Uncertainty-Weighted Multi-Task Loss Function

To train both heads simultaneously without manual hyperparameter tuning, we implement **Kendall et al. Homoscedastic Task Uncertainty Weighting**:

$$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{diffusion}} + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{recon}} + \frac{1}{\sigma_3^2} \mathcal{L}_{\text{anomaly\_BCE}} + \log(\sigma_1 \sigma_2 \sigma_3)$$

where $\sigma_1, \sigma_2, \sigma_3$ are learnable variance parameters updated automatically via gradient descent.

---

## 4. Empirical Performance Gains vs. Baseline Models (highD Benchmark)

| Model Architecture | Input Features | Trajectory ADE ($T_{\text{pred}}=3s$) | Trajectory FDE ($T_{\text{pred}}=3s$) | Anomaly F1-Score | Anomaly AUC-ROC | Inference Latency (Batch / Single) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM-Autoencoder** | $(x, y)$ | 2.834 m | 5.639 m | 0.3099 | 0.5532 | 0.34 ms |
| **ST-GCN** | $(x, y, v)$ | 2.500 m | 4.912 m | 0.2701 | 0.4339 | 0.20 ms |
| **PatchTST** | $(x, y, v, a)$ | 0.832 m | 1.508 m | 0.6383 | 0.6186 | 0.62 ms |
| **AA-STGAT (Base Paper)** | $(x, y, v, \theta)$ | 0.925 m | 1.829 m | 0.1667 | 0.4952 | 1.83 ms |
| **MT-STGTD (Proposed)** | **$7D$ Multi-Modal** | **0.588 m (-36.4%)** | **1.155 m (-36.9%)** | **0.6791 (+0.512)** | **0.9313 (+0.436)** | **0.77 ms / 5.51 ms** |

---

## 5. Tensor Shapes & Layer Specifications

```
Input Telemetry:         [Batch, N_vehicles, T_obs=20, Dim=7]
  ├── Continuous Embedding  --> [Batch, N_vehicles, 20, 128]
  ├── Dynamic GAT (V2V)     --> [Batch, N_vehicles, 20, 128]
  ├── Patchification (P=4,S=2)--> [Batch, N_vehicles, M=9, 128]
  ├── Transformer Encoder   --> [Batch, N_vehicles, 9, 256] (Z_ST)
  │
  ├── Head A (Diffusion)    --> Trajectory Forecast: [Batch, N_vehicles, T_pred=30, 2] (x, y)
  └── Head B (Masked Recon) --> Anomaly Score:       [Batch, N_vehicles, 1] (S_anomaly ∈ [0, 1])
```
