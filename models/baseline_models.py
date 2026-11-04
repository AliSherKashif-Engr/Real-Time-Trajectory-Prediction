"""
Baseline Deep Learning Architectures for Trajectory Prediction & Anomaly Detection
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)

Implements competitive SOTA baselines for comparative benchmarking against MT-STGTD:
1. LSTM-Autoencoder: 2D coordinates (x, y) with Seq2Seq recurrent architecture
2. ST-GCN: 3D kinematics (x, y, v) with Spatial Graph Convolutions + Temporal Conv1D
3. PatchTST: 4D kinematics (x, y, v, a) with Patchified Temporal Transformer
4. AA-STGAT: 4D kinematics (x, y, v, theta) with Graph Attention & decoupled heads (Base Paper)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Baseline 1: LSTM-Autoencoder (2D Coordinates)
# ============================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, in_features=2, hidden_dim=64, num_layers=2, pred_steps=30):
        super().__init__()
        self.in_features = in_features
        self.pred_steps = pred_steps
        
        self.encoder_lstm = nn.LSTM(
            input_size=in_features, hidden_size=hidden_dim, 
            num_layers=num_layers, batch_first=True
        )
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pred_steps * 2)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, dist_matrix=None):
        # x: [B, N, T_obs, 7] -> isolate (x, y) coordinates for target vehicle and neighbors
        B, N, T, C = x.shape
        x_2d = x[:, :, :, :2].contiguous().view(B * N, T, 2)
        
        _, (h_n, _) = self.encoder_lstm(x_2d) # h_n: [num_layers, B*N, hidden_dim]
        h_last = h_n[-1] # [B*N, hidden_dim]
        
        # Trajectory forecast
        pred_delta = self.traj_head(h_last).view(B, N, self.pred_steps, 2)
        # Physics offset from last position
        x0 = x[:, :, -1, :2].unsqueeze(2) # [B, N, 1, 2]
        pred_traj = x0 + pred_delta
        
        # Anomaly prediction
        anomaly_score = self.anomaly_head(h_last).view(B, N, 1)
        
        return {
            "pred_traj": pred_traj,
            "anomaly_score": anomaly_score,
            "latent": h_last.view(B, N, -1)
        }

    def compute_loss(self, outputs, target_traj, target_anomaly):
        loss_traj = F.mse_loss(outputs["pred_traj"], target_traj)
        cls_weight = torch.where(target_anomaly == 1.0, torch.tensor(5.0, device=target_anomaly.device), torch.tensor(1.0, device=target_anomaly.device))
        loss_cls = F.binary_cross_entropy(outputs["anomaly_score"], target_anomaly, weight=cls_weight)
        total_loss = loss_traj + 0.5 * loss_cls
        return total_loss, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


# ============================================================================
# Baseline 2: ST-GCN (3D Kinematics: x, y, v)
# ============================================================================
class SpatialGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [B, T, N, C], adj: [B, T, N, N]
        support = self.linear(x) # [B, T, N, out_features]
        out = torch.matmul(adj, support) # [B, T, N, out_features]
        return F.relu(out)

class STGCNModel(nn.Module):
    def __init__(self, in_features=3, hidden_dim=64, pred_steps=30):
        super().__init__()
        self.pred_steps = pred_steps
        
        self.input_proj = nn.Linear(3, hidden_dim)
        self.gcn1 = SpatialGraphConv(hidden_dim, hidden_dim)
        self.tconv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pred_steps * 2)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, dist_matrix=None):
        # x: [B, N, T, 7] -> isolate (x, y, v) [B, N, T, 3]
        B, N, T, C = x.shape
        x_3d = torch.cat([x[:, :, :, :2], x[:, :, :, 2:3]], dim=-1) # [B, N, T, 3]
        x_perm = x_3d.permute(0, 2, 1, 3) # [B, T, N, 3]
        
        # Build normalized adjacency from distance matrix
        if dist_matrix is not None:
            adj = torch.exp(-dist_matrix / 50.0) # [B, T, N, N]
            adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            adj = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, T, N, N)
            
        h = self.input_proj(x_perm)
        h_spatial = self.gcn1(h, adj) # [B, T, N, hidden_dim]
        
        # Temporal convolution over T
        h_temp = h_spatial.permute(0, 2, 3, 1).contiguous().view(B * N, -1, T) # [B*N, hidden_dim, T]
        h_conv = F.relu(self.tconv1(h_temp)) # [B*N, hidden_dim, T]
        h_pool = h_conv.mean(dim=-1).view(B, N, -1) # [B, N, hidden_dim]
        
        # Prior offset from constant velocity
        v0 = x[:, :, -1, 2].unsqueeze(-1)
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * 0.1
        prior_x = x0 + v0 * t_steps.unsqueeze(0).unsqueeze(0)
        prior_y = y0.expand_as(prior_x)
        prior_traj = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_head(h_pool).view(B, N, self.pred_steps, 2)
        pred_traj = prior_traj + delta
        anomaly_score = self.anomaly_head(h_pool)
        
        return {
            "pred_traj": pred_traj,
            "anomaly_score": anomaly_score,
            "latent": h_pool
        }

    def compute_loss(self, outputs, target_traj, target_anomaly):
        loss_traj = F.mse_loss(outputs["pred_traj"], target_traj)
        cls_weight = torch.where(target_anomaly == 1.0, torch.tensor(5.0, device=target_anomaly.device), torch.tensor(1.0, device=target_anomaly.device))
        loss_cls = F.binary_cross_entropy(outputs["anomaly_score"], target_anomaly, weight=cls_weight)
        total_loss = loss_traj + 0.5 * loss_cls
        return total_loss, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


# ============================================================================
# Baseline 3: PatchTST / Spatio-Temporal Transformer (4D: x, y, v, a)
# ============================================================================
class PatchTSTModel(nn.Module):
    def __init__(self, in_features=4, d_model=128, patch_len=4, stride=2, num_layers=2, pred_steps=30):
        super().__init__()
        self.pred_steps = pred_steps
        self.patch_len = patch_len
        self.stride = stride
        
        self.patch_proj = nn.Linear(patch_len * 4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, pred_steps * 2)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, dist_matrix=None):
        # x: [B, N, T, 7] -> isolate (x, y, v, ax) [B, N, T, 4]
        B, N, T, C = x.shape
        x_4d = x[:, :, :, :4] # [B, N, T, 4]
        
        # Patchify across T dimension
        patches = x_4d.unfold(dimension=2, size=self.patch_len, step=self.stride) # [B, N, M, 4, P]
        B, N, M, D, P = patches.shape
        patches = patches.permute(0, 1, 2, 4, 3).contiguous().view(B * N, M, P * D)
        
        patch_emb = self.patch_proj(patches)
        z = self.transformer(patch_emb) # [B*N, M, d_model]
        z_global = z.mean(dim=1).view(B, N, -1) # [B, N, d_model]
        
        # Prior offset
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        v0 = x[:, :, -1, 2].unsqueeze(-1)
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * 0.1
        prior_x = x0 + v0 * t_steps.unsqueeze(0).unsqueeze(0)
        prior_y = y0.expand_as(prior_x)
        prior_traj = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_head(z_global).view(B, N, self.pred_steps, 2)
        pred_traj = prior_traj + delta
        anomaly_score = self.anomaly_head(z_global)
        
        return {
            "pred_traj": pred_traj,
            "anomaly_score": anomaly_score,
            "latent": z_global
        }

    def compute_loss(self, outputs, target_traj, target_anomaly):
        loss_traj = F.mse_loss(outputs["pred_traj"], target_traj)
        cls_weight = torch.where(target_anomaly == 1.0, torch.tensor(5.0, device=target_anomaly.device), torch.tensor(1.0, device=target_anomaly.device))
        loss_cls = F.binary_cross_entropy(outputs["anomaly_score"], target_anomaly, weight=cls_weight)
        total_loss = loss_traj + 0.5 * loss_cls
        return total_loss, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


# ============================================================================
# Baseline 4: AA-STGAT (Base Paper Benchmark - 4D: x, y, v, theta)
# ============================================================================
class AASTGATModel(nn.Module):
    def __init__(self, in_features=4, d_model=128, num_layers=2, pred_steps=30):
        super().__init__()
        self.pred_steps = pred_steps
        
        self.input_proj = nn.Linear(4, d_model)
        self.spatial_gat = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.1
        )
        self.temporal_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, pred_steps * 2)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, dist_matrix=None):
        # x: [B, N, T, 7] -> isolate (x, y, v, theta)
        B, N, T, C = x.shape
        x_4d = torch.cat([x[:, :, :, :3], x[:, :, :, 5:6]], dim=-1) # [B, N, T, 4]
        
        h = self.input_proj(x_4d) # [B, N, T, d_model]
        
        # Spatial Multihead Attention across N at each time step
        h_spatial_list = []
        for t in range(T):
            h_t = h[:, :, t, :] # [B, N, d_model]
            attn_out, _ = self.spatial_gat(h_t, h_t, h_t)
            h_spatial_list.append(attn_out.unsqueeze(2))
        h_spatial = torch.cat(h_spatial_list, dim=2) # [B, N, T, d_model]
        
        # Temporal Attention across T for each vehicle
        h_flat = h_spatial.view(B * N, T, -1)
        z_temp = self.temporal_trans(h_flat) # [B*N, T, d_model]
        z_global = z_temp.mean(dim=1).view(B, N, -1) # [B, N, d_model]
        
        # Prior offset
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        v0 = x[:, :, -1, 2].unsqueeze(-1)
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * 0.1
        prior_x = x0 + v0 * t_steps.unsqueeze(0).unsqueeze(0)
        prior_y = y0.expand_as(prior_x)
        prior_traj = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_head(z_global).view(B, N, self.pred_steps, 2)
        pred_traj = prior_traj + delta
        anomaly_score = self.anomaly_head(z_global)
        
        return {
            "pred_traj": pred_traj,
            "anomaly_score": anomaly_score,
            "latent": z_global
        }

    def compute_loss(self, outputs, target_traj, target_anomaly):
        loss_traj = F.mse_loss(outputs["pred_traj"], target_traj)
        cls_weight = torch.where(target_anomaly == 1.0, torch.tensor(5.0, device=target_anomaly.device), torch.tensor(1.0, device=target_anomaly.device))
        loss_cls = F.binary_cross_entropy(outputs["anomaly_score"], target_anomaly, weight=cls_weight)
        total_loss = loss_traj + 0.5 * loss_cls
        return total_loss, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}
