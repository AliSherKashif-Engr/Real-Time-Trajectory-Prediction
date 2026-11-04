"""
MT-STGTD: Multi-Task Spatio-Temporal Graph Transformer with Diffusion Refinement 
          & Reconstruction-Guided Anomaly Detection
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSpatialGAT(nn.Module):
    """Multi-Head Dynamic Graph Attention Network for V2V interaction modeling."""
    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.fc_q = nn.Linear(in_features, out_features)
        self.fc_k = nn.Linear(in_features, out_features)
        self.fc_v = nn.Linear(in_features, out_features)
        self.out_proj = nn.Linear(out_features, out_features)
        
    def forward(self, x, dist_matrix=None):
        # x shape: [B, N, T, C]
        B, N, T, C = x.shape
        x_perm = x.permute(0, 2, 1, 3) # [B, T, N, C]
        
        Q = self.fc_q(x_perm).view(B, T, N, self.num_heads, self.head_dim).transpose(2, 3)
        K = self.fc_k(x_perm).view(B, T, N, self.num_heads, self.head_dim).transpose(2, 3)
        V = self.fc_v(x_perm).view(B, T, N, self.num_heads, self.head_dim).transpose(2, 3)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, T, heads, N, N]
        
        if dist_matrix is not None:
            # Apply dynamic distance decay mask
            dist_weight = torch.exp(-dist_matrix / 50.0).unsqueeze(2) # [B, T, 1, N, N]
            scores = scores + torch.log(dist_weight + 1e-8)
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V) # [B, T, heads, N, head_dim]
        out = out.transpose(2, 3).contiguous().view(B, T, N, -1)
        out = self.out_proj(out).permute(0, 2, 1, 3) # [B, N, T, C_out]
        return out

class PatchifiedSTEncoder(nn.Module):
    """Patchified Spatio-Temporal Transformer Encoder (PatchTST style for low complexity)."""
    def __init__(self, d_model=128, patch_len=4, stride=2, num_layers=2, num_heads=4):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len * d_model, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: [B, N, T, d_model]
        B, N, T, D = x.shape
        # Unfold along T dimension into patches
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride) # [B, N, M, D, P]
        B, N, M, D, P = patches.shape
        patches = patches.permute(0, 1, 2, 4, 3).contiguous().view(B * N, M, P * D)
        
        patch_emb = self.patch_proj(patches) # [B*N, M, d_model]
        out = self.transformer(patch_emb)     # [B*N, M, d_model]
        out = out.view(B, N, M, D)
        return out

class MTSTGTDModel(nn.Module):
    """
    Unified Multi-Task Spatio-Temporal Graph Transformer (MT-STGTD).
    Branch A: Future Trajectory Prediction (x, y coordinates)
    Branch B: Telemetry Masked Reconstruction & Anomaly Scoring
    """
    def __init__(self, in_features=7, d_model=128, pred_steps=30, obs_steps=20):
        super().__init__()
        self.pred_steps = pred_steps
        self.obs_steps = obs_steps
        
        # 1. Feature Embedding
        self.input_proj = nn.Linear(in_features, d_model)
        
        # 2. Dynamic Spatial GAT
        self.gat = DynamicSpatialGAT(in_features=d_model, out_features=d_model, num_heads=4)
        
        # 3. Patchified Spatio-Temporal Transformer Encoder
        self.encoder = PatchifiedSTEncoder(d_model=d_model, patch_len=4, stride=2, num_layers=2)
        
        # 4. Multi-Task Heads
        # Head A: Kinematic Residual Trajectory Decoder
        self.traj_residual_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, pred_steps * 2)
        )
        # Small weight initialization for residual branch so it starts near physics prior
        nn.init.normal_(self.traj_residual_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.traj_residual_head[-1].bias)
        
        # Head B: Telemetry Reconstruction & Composite Energy Anomaly Scoring
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, in_features)
        )
        # Learnable positive scaling weights for energy combination
        self.energy_weight_recon = nn.Parameter(torch.tensor(1.0))
        self.energy_weight_traj = nn.Parameter(torch.tensor(0.5))
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Learnable uncertainty weights for loss balancing (Kendall et al.)
        self.log_var_traj = nn.Parameter(torch.tensor(0.0))
        self.log_var_recon = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, dist_matrix=None):
        # x: [B, N, T_obs, 7]
        B, N, T, C = x.shape
        
        # Embedding + GAT Spatial Pooling
        h_emb = F.relu(self.input_proj(x))
        h_spatial = self.gat(h_emb, dist_matrix)
        
        # Patchified Temporal Encoder
        z_st = self.encoder(h_spatial) # [B, N, M, d_model]
        z_global = z_st.mean(dim=2)     # Global pooled representation [B, N, d_model]
        
        # 1. Physics-Guided Kinematic Prior (Constant Velocity in Ego Frame)
        # Include initial position offset x0, y0 at t=T_obs-1
        x0 = x[:, :, -1, 0].unsqueeze(-1) # [B, N, 1]
        y0 = x[:, :, -1, 1].unsqueeze(-1) # [B, N, 1]
        
        v0 = x[:, :, -1, 2] # Instantaneous scalar speed at t=T_obs-1
        heading0 = x[:, :, -1, 5] # Relative heading at t=T_obs-1
        vx_ego = v0 * torch.cos(heading0)
        vy_ego = v0 * torch.sin(heading0)
        
        dt = 0.1 # 10 Hz sampling rate
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * dt # [pred_steps]
        prior_x = x0 + vx_ego.unsqueeze(-1) * t_steps.unsqueeze(0).unsqueeze(0) # [B, N, pred_steps]
        prior_y = y0 + vy_ego.unsqueeze(-1) * t_steps.unsqueeze(0).unsqueeze(0) # [B, N, pred_steps]
        prior_traj = torch.stack([prior_x, prior_y], dim=-1) # [B, N, pred_steps, 2]
        
        # Head A: Residual Trajectory Refinement
        delta_traj = self.traj_residual_head(z_global).view(B, N, self.pred_steps, 2)
        pred_traj = prior_traj + delta_traj
        
        # Head B: Telemetry Reconstruction
        target_telemetry_last = x[:, :, -1, :] # [B, N, 7]
        recon_telemetry = self.recon_head(z_global) # [B, N, 7]
        
        # Feature Reconstruction Energy: [B, N, 1]
        recon_energy = torch.mean((recon_telemetry - target_telemetry_last) ** 2, dim=-1, keepdim=True)
        
        # Trajectory Deviation Energy from Prior: [B, N, 1]
        traj_energy = torch.mean(torch.sum(delta_traj ** 2, dim=-1), dim=-1, keepdim=True)
        
        # Composite Energy Anomaly Score (Section 3.5):
        # S_anomaly = sigmoid(w_recon * E_recon + w_traj * E_traj + classifier(z))
        w_r = F.softplus(self.energy_weight_recon)
        w_t = F.softplus(self.energy_weight_traj)
        cls_logits = self.anomaly_classifier(z_global)
        
        composite_energy = w_r * recon_energy + w_t * traj_energy + cls_logits
        anomaly_score = torch.sigmoid(composite_energy)
        
        return {
            "pred_traj": pred_traj,
            "prior_traj": prior_traj,
            "delta_traj": delta_traj,
            "recon_telemetry": recon_telemetry,
            "recon_energy": recon_energy,
            "traj_energy": traj_energy,
            "anomaly_score": anomaly_score,
            "latent": z_global
        }

    def compute_loss(self, outputs, target_traj, target_telemetry, target_anomaly):
        """Dynamic Uncertainty-Weighted Multi-Task Loss."""
        loss_traj = F.mse_loss(outputs["pred_traj"], target_traj)
        loss_recon = F.mse_loss(outputs["recon_telemetry"], target_telemetry)
        
        # Binary cross entropy with positive weighting for class imbalance handling
        cls_weight = torch.where(target_anomaly == 1.0, torch.tensor(5.0, device=target_anomaly.device), torch.tensor(1.0, device=target_anomaly.device))
        loss_cls = F.binary_cross_entropy(
            outputs["anomaly_score"], target_anomaly, weight=cls_weight
        )
        
        # Kendall uncertainty weighting
        precision_traj = torch.exp(-self.log_var_traj)
        precision_recon = torch.exp(-self.log_var_recon)
        precision_cls = torch.exp(-self.log_var_cls)
        
        total_loss = (
            precision_traj * loss_traj + 0.5 * self.log_var_traj +
            precision_recon * loss_recon + 0.5 * self.log_var_recon +
            precision_cls * loss_cls + 0.5 * self.log_var_cls
        )
        return total_loss, {
            "loss_traj": loss_traj.item(),
            "loss_recon": loss_recon.item(),
            "loss_cls": loss_cls.item()
        }

if __name__ == "__main__":
    print("Testing MT-STGTD PyTorch Architecture...")
    model = MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20)
    dummy_input = torch.randn(4, 10, 20, 7) # Batch=4, Vehicles=10, Obs_steps=20, Dim=7
    out = model(dummy_input)
    print("Output Trajectory Shape:", out["pred_traj"].shape)       # Expected: [4, 10, 30, 2]
    print("Output Anomaly Score Shape:", out["anomaly_score"].shape) # Expected: [4, 10, 1]
    print("Model forward pass successful!")
