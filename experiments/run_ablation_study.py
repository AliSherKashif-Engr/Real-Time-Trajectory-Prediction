"""
Component-Wise Architectural Ablation Study for MT-STGTD (Paper 01)
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)

Rigorously measures the incremental contribution of each component:
  M1: Baseline (2D coordinates, no GAT, standard MLP)
  M2: + 7D Multi-Modal Telemetry Vector
  M3: + Dynamic Spatial Graph Attention Network (D-GAT)
  M4: + Patchified Temporal Transformer (Patch-STGT)
  M5: + Kinematic Residual Trajectory Prior
  M6: Full MT-STGTD (+ Composite Energy Head & Kendall Uncertainty Loss)

Outputs Table 4.3 formatted for IEEE Transactions on Intelligent Transportation Systems.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mt_stgtd_model import MTSTGTDModel, DynamicSpatialGAT, PatchifiedSTEncoder
from experiments.highd_dataset_loader import HighDDataPreprocessor

# ---------------------------------------------------------------------------
# Ablation Variants
# ---------------------------------------------------------------------------

class M1_Baseline(nn.Module):
    """M1: 2D Coordinates, Linear Projection, Simple Temporal Pooling, Static Loss"""
    def __init__(self, pred_steps=30, d_model=128):
        super().__init__()
        self.pred_steps = pred_steps
        self.proj = nn.Linear(2, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.traj_head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, pred_steps * 2))
        self.cls_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, dist_matrix=None):
        B, N, T, C = x.shape
        x_2d = x[:, :, :, :2].contiguous().view(B * N, T, 2)
        h = self.proj(x_2d)
        _, (h_n, _) = self.lstm(h)
        z = h_n[-1].view(B, N, -1)
        
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        prior_x = x0.expand(B, N, self.pred_steps)
        prior_y = y0.expand(B, N, self.pred_steps)
        prior = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_head(z).view(B, N, self.pred_steps, 2)
        pred_traj = prior + delta
        anomaly_score = self.cls_head(z)
        return {"pred_traj": pred_traj, "anomaly_score": anomaly_score, "latent": z}

    def compute_loss(self, out, target_y, target_telemetry, target_anomaly):
        loss_traj = F.mse_loss(out["pred_traj"], target_y)
        loss_cls = F.binary_cross_entropy(out["anomaly_score"], target_anomaly)
        return loss_traj + 0.5 * loss_cls, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


class M2_7D_Telemetry(nn.Module):
    """M2: + 7D Multi-Modal Telemetry Vector (No GAT, Identity Graph)"""
    def __init__(self, pred_steps=30, d_model=128):
        super().__init__()
        self.pred_steps = pred_steps
        self.proj = nn.Linear(7, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.traj_head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, pred_steps * 2))
        self.cls_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, dist_matrix=None):
        B, N, T, C = x.shape
        x_flat = x.view(B * N, T, 7)
        h = self.proj(x_flat)
        _, (h_n, _) = self.lstm(h)
        z = h_n[-1].view(B, N, -1)
        
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        v0 = x[:, :, -1, 2].unsqueeze(-1)
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * 0.1
        prior_x = x0 + v0 * t_steps.unsqueeze(0).unsqueeze(0)
        prior_y = y0.expand_as(prior_x)
        prior = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_head(z).view(B, N, self.pred_steps, 2)
        pred_traj = prior + delta
        anomaly_score = self.cls_head(z)
        return {"pred_traj": pred_traj, "anomaly_score": anomaly_score, "latent": z}

    def compute_loss(self, out, target_y, target_telemetry, target_anomaly):
        loss_traj = F.mse_loss(out["pred_traj"], target_y)
        loss_cls = F.binary_cross_entropy(out["anomaly_score"], target_anomaly)
        return loss_traj + 0.5 * loss_cls, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


class M3_DGAT(nn.Module):
    """M3: + Dynamic Spatial Graph Attention (D-GAT)"""
    def __init__(self, pred_steps=30, d_model=128):
        super().__init__()
        self.pred_steps = pred_steps
        self.input_proj = nn.Linear(7, d_model)
        self.gat = DynamicSpatialGAT(in_features=d_model, out_features=d_model, num_heads=4)
        self.temporal_trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, batch_first=True),
            num_layers=2
        )
        self.traj_head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, pred_steps * 2))
        self.cls_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, dist_matrix=None):
        B, N, T, C = x.shape
        h = F.relu(self.input_proj(x))
        h_spatial = self.gat(h, dist_matrix)
        h_flat = h_spatial.reshape(B * N, T, -1)
        z = self.temporal_trans(h_flat).mean(dim=1).view(B, N, -1)
        
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        v0 = x[:, :, -1, 2].unsqueeze(-1)
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * 0.1
        prior_x = x0 + v0 * t_steps.unsqueeze(0).unsqueeze(0)
        prior_y = y0.expand_as(prior_x)
        prior = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_head(z).view(B, N, self.pred_steps, 2)
        pred_traj = prior + delta
        anomaly_score = self.cls_head(z)
        return {"pred_traj": pred_traj, "anomaly_score": anomaly_score, "latent": z}

    def compute_loss(self, out, target_y, target_telemetry, target_anomaly):
        loss_traj = F.mse_loss(out["pred_traj"], target_y)
        loss_cls = F.binary_cross_entropy(out["anomaly_score"], target_anomaly)
        return loss_traj + 0.5 * loss_cls, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


class M4_PatchSTGT(nn.Module):
    """M4: + Patchified Temporal Encoder (Patch-STGT)"""
    def __init__(self, pred_steps=30, d_model=128):
        super().__init__()
        self.pred_steps = pred_steps
        self.input_proj = nn.Linear(7, d_model)
        self.gat = DynamicSpatialGAT(in_features=d_model, out_features=d_model, num_heads=4)
        self.encoder = PatchifiedSTEncoder(d_model=d_model, patch_len=4, stride=2, num_layers=2)
        self.traj_head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, pred_steps * 2))
        self.cls_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, dist_matrix=None):
        B, N, T, C = x.shape
        h = F.relu(self.input_proj(x))
        h_spatial = self.gat(h, dist_matrix)
        z_st = self.encoder(h_spatial)
        z = z_st.mean(dim=2)
        
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        v0 = x[:, :, -1, 2].unsqueeze(-1)
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * 0.1
        prior_x = x0 + v0 * t_steps.unsqueeze(0).unsqueeze(0)
        prior_y = y0.expand_as(prior_x)
        prior = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_head(z).view(B, N, self.pred_steps, 2)
        pred_traj = prior + delta
        anomaly_score = self.cls_head(z)
        return {"pred_traj": pred_traj, "anomaly_score": anomaly_score, "latent": z}

    def compute_loss(self, out, target_y, target_telemetry, target_anomaly):
        loss_traj = F.mse_loss(out["pred_traj"], target_y)
        loss_cls = F.binary_cross_entropy(out["anomaly_score"], target_anomaly)
        return loss_traj + 0.5 * loss_cls, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


class M5_KinematicResidual(nn.Module):
    """M5: + Kinematic Residual Refinement Head (Static Loss)"""
    def __init__(self, pred_steps=30, d_model=128):
        super().__init__()
        self.pred_steps = pred_steps
        self.input_proj = nn.Linear(7, d_model)
        self.gat = DynamicSpatialGAT(in_features=d_model, out_features=d_model, num_heads=4)
        self.encoder = PatchifiedSTEncoder(d_model=d_model, patch_len=4, stride=2, num_layers=2)
        self.traj_residual_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, pred_steps * 2)
        )
        self.cls_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, dist_matrix=None):
        B, N, T, C = x.shape
        h = F.relu(self.input_proj(x))
        h_spatial = self.gat(h, dist_matrix)
        z = self.encoder(h_spatial).mean(dim=2)
        
        x0 = x[:, :, -1, 0].unsqueeze(-1)
        y0 = x[:, :, -1, 1].unsqueeze(-1)
        v0 = x[:, :, -1, 2]
        heading0 = x[:, :, -1, 5]
        vx_ego = v0 * torch.cos(heading0)
        vy_ego = v0 * torch.sin(heading0)
        
        t_steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype) * 0.1
        prior_x = x0 + vx_ego.unsqueeze(-1) * t_steps.unsqueeze(0).unsqueeze(0)
        prior_y = y0 + vy_ego.unsqueeze(-1) * t_steps.unsqueeze(0).unsqueeze(0)
        prior = torch.stack([prior_x, prior_y], dim=-1)
        
        delta = self.traj_residual_head(z).view(B, N, self.pred_steps, 2)
        pred_traj = prior + delta
        anomaly_score = self.cls_head(z)
        return {"pred_traj": pred_traj, "anomaly_score": anomaly_score, "latent": z}

    def compute_loss(self, out, target_y, target_telemetry, target_anomaly):
        loss_traj = F.mse_loss(out["pred_traj"], target_y)
        loss_cls = F.binary_cross_entropy(out["anomaly_score"], target_anomaly)
        return loss_traj + 0.5 * loss_cls, {"loss_traj": loss_traj.item(), "loss_cls": loss_cls.item()}


# ---------------------------------------------------------------------------
# Runner for Ablation Study
# ---------------------------------------------------------------------------

def run_ablation_experiments():
    print("="*95)
    print("  Paper 01: Component-Wise Architectural Ablation Study")
    print("  Empirical Verification across M1 -> M6 on highD Benchmark")
    print("="*95)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Platform: {device}")
    
    preproc = HighDDataPreprocessor(data_dir="data", ego_centric=True)
    train_ds = preproc.load_dataset_split([1, 2], cache_name="train_highd_quick")
    val_ds = preproc.load_dataset_split([3], cache_name="val_highd_quick")
    test_ds = preproc.load_dataset_split([4], cache_name="test_highd_quick")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    variants = [
        ("M1: Baseline ST-GNN (2D)", M1_Baseline()),
        ("M2: + 7D Telemetry Vector", M2_7D_Telemetry()),
        ("M3: + Dynamic Spatial GAT", M3_DGAT()),
        ("M4: + Patchified Encoder", M4_PatchSTGT()),
        ("M5: + Kinematic Residual Prior", M5_KinematicResidual()),
        ("M6: Full MT-STGTD (+ Energy Head & Uncertainty Loss)", MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20))
    ]
    
    ablation_results = {}
    
    for name, model in variants:
        print(f"\nEvaluating Ablation Variant: [{name}]...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.to(device)
        
        # Train for 5 epochs
        for epoch in range(1, 6):
            model.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                dist_mat = batch["dist_matrix"].to(device)
                y = batch["y"].to(device)
                labels = batch["anomaly_label"].to(device)
                
                optimizer.zero_grad()
                out = model(x, dist_mat)
                target_telemetry = x[:, :, -1, :]
                loss, _ = model.compute_loss(out, y, target_telemetry, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
        # Evaluate on test split
        model.eval()
        all_preds, all_targets, all_labels = [], [], []
        all_scores = []
        latencies = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                dist_mat = batch["dist_matrix"].to(device)
                y = batch["y"].to(device)
                labels = batch["anomaly_label"].to(device)
                
                t0 = time.perf_counter()
                out = model(x, dist_mat)
                latencies.append((time.perf_counter() - t0) * 1000.0 / len(x))
                
                all_preds.append(out["pred_traj"].cpu())
                all_targets.append(y.cpu())
                all_labels.append(labels.cpu())
                all_scores.append(out["anomaly_score"][:, 0, 0].cpu().numpy())
                
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_labels = torch.cat(all_labels, dim=0)
        flat_scores = np.concatenate(all_scores)
        flat_labels = cat_labels[:, 0, 0].numpy()
        
        diff = cat_preds[:, 0, :, :] - cat_targets[:, 0, :, :]
        euc_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        normal_mask = (cat_labels[:, 0, 0] == 0)
        norm_dist = euc_dist[normal_mask] if normal_mask.sum() > 0 else euc_dist
        
        ade_1s = float(norm_dist[:, :10].mean().item())
        ade_3s = float(norm_dist[:, :30].mean().item())
        fde_3s = float(norm_dist[:, 29].mean().item())
        
        auc_roc = float(roc_auc_score(flat_labels, flat_scores))
        precs, recs, threshs = precision_recall_curve(flat_labels, flat_scores)
        f1_arr = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-8)
        best_f1 = float(np.max(f1_arr)) if len(f1_arr) > 0 else 0.0
        
        avg_lat = float(np.mean(latencies))
        total_p = sum(p.numel() for p in model.parameters())
        
        res = {
            "ade_1s": round(ade_1s, 3),
            "ade_3s": round(ade_3s, 3),
            "fde_3s": round(fde_3s, 3),
            "f1_score": round(best_f1, 4),
            "auc_roc": round(auc_roc, 4),
            "latency_ms": round(avg_lat, 2),
            "params": f"{total_p/1e6:.2f}M"
        }
        ablation_results[name] = res
        print(f"  --> {name}: ADE(3s)={res['ade_3s']}m, F1={res['f1_score']}, AUC={res['auc_roc']}, Latency={res['latency_ms']}ms")
        
    print("\n" + "="*105)
    print("Table 4.3: Component-Wise Ablation Study Results on highD Benchmark")
    print("="*105)
    print(f"{'Ablation Variant':<45} | {'ADE (1s)':<8} | {'ADE (3s)':<8} | {'FDE (3s)':<8} | {'F1-Score':<8} | {'AUC-ROC':<8} | {'Latency':<8}")
    print("="*105)
    for v_name, m in ablation_results.items():
        print(f"{v_name:<45} | {m['ade_1s']:<8.3f} | {m['ade_3s']:<8.3f} | {m['fde_3s']:<8.3f} | {m['f1_score']:<8.4f} | {m['auc_roc']:<8.4f} | {m['latency_ms']:<6.2f}ms")
    print("="*105)
    
    out_file = "experiments/ablation_study_results.json"
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=4)
    print(f"\nAblation study results exported to '{out_file}'.")
    return ablation_results

if __name__ == "__main__":
    run_ablation_experiments()
