"""
Training and Empirical Evaluation Suite for MT-STGTD on Authentic highD Dataset
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)

Trains MT-STGTD on authentic naturalistic vehicle trajectories from highD recordings,
evaluates joint trajectory forecasting (ADE/FDE) and anomaly detection (F1, AUC-ROC),
and benchmarks real-time inference latency.
"""

import os
import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# Import project components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mt_stgtd_model import MTSTGTDModel
from experiments.highd_dataset_loader import HighDDataPreprocessor

from sklearn.metrics import precision_recall_curve

def compute_displacement_errors(pred_traj, target_traj, mask, anomaly_labels=None):
    """
    Computes Average Displacement Error (ADE) and Final Displacement Error (FDE) in meters.
    pred_traj: [B, N, T_pred, 2]
    target_traj: [B, N, T_pred, 2]
    mask: [B, N] (1 for real vehicles)
    """
    diff = pred_traj - target_traj
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1)) # [B, N, T_pred]
    
    # 1. Target ego vehicle (slot 0) errors
    ego_dist = dist[:, 0, :] # [B, T_pred]
    ade_1s = float(ego_dist[:, :10].mean().item())
    ade_2s = float(ego_dist[:, :20].mean().item())
    ade_3s = float(ego_dist[:, :30].mean().item())
    fde_3s = float(ego_dist[:, 29].mean().item())
    
    # 2. Target vehicle on nominal highway driving (excluding injected anomalies)
    normal_ade_3s = ade_3s
    normal_fde_3s = fde_3s
    if anomaly_labels is not None:
        normal_mask = (anomaly_labels[:, 0, 0] == 0)
        if normal_mask.sum() > 0:
            norm_ego_dist = ego_dist[normal_mask]
            normal_ade_1s = float(norm_ego_dist[:, :10].mean().item())
            normal_ade_2s = float(norm_ego_dist[:, :20].mean().item())
            normal_ade_3s = float(norm_ego_dist[:, :30].mean().item())
            normal_fde_3s = float(norm_ego_dist[:, 29].mean().item())
    
    return {
        "ade_1s": round(ade_1s, 3),
        "ade_2s": round(ade_2s, 3),
        "ade_3s": round(ade_3s, 3),
        "fde_3s": round(fde_3s, 3),
        "normal_ade_1s": round(normal_ade_1s, 3),
        "normal_ade_2s": round(normal_ade_2s, 3),
        "normal_ade_3s": round(normal_ade_3s, 3),
        "normal_fde_3s": round(normal_fde_3s, 3)
    }

def evaluate_model(model, data_loader, device="cpu", eval_threshold=None):
    """
    Evaluates model on validation or test DataLoader.
    Computes ADE, FDE, Anomaly Classification Metrics (AUC-ROC, F1), and Latency.
    """
    model.eval()
    all_pred_trajs, all_target_trajs, all_masks = [], [], []
    all_anomaly_scores, all_anomaly_labels = [], []
    inference_times = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            dist_mat = batch["dist_matrix"].to(device)
            y = batch["y"].to(device)
            anomaly_label = batch["anomaly_label"].to(device)
            mask = batch["mask"].to(device)
            
            # Benchmark latency
            t0 = time.perf_counter()
            outputs = model(x, dist_mat)
            t_infer = (time.perf_counter() - t0) * 1000.0 # ms
            inference_times.append(t_infer / len(x)) # ms per sample
            
            all_pred_trajs.append(outputs["pred_traj"].cpu())
            all_target_trajs.append(y.cpu())
            all_masks.append(mask.cpu())
            all_anomaly_labels.append(anomaly_label.cpu())
            
            # Target vehicle (slot 0) anomaly evaluation
            all_anomaly_scores.append(outputs["anomaly_score"][:, 0, 0].cpu().numpy())
            
    cat_preds = torch.cat(all_pred_trajs, dim=0)
    cat_targets = torch.cat(all_target_trajs, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)
    cat_labels = torch.cat(all_anomaly_labels, dim=0)
    
    disp_errors = compute_displacement_errors(cat_preds, cat_targets, cat_masks, cat_labels)
    
    # Anomaly detection metrics
    flat_scores = np.concatenate(all_anomaly_scores)
    flat_labels = cat_labels[:, 0, 0].numpy()
    
    if len(np.unique(flat_labels)) > 1:
        auc_roc = float(roc_auc_score(flat_labels, flat_scores))
        pr_auc = float(average_precision_score(flat_labels, flat_scores))
        
        # Determine optimal F1 threshold from precision-recall curve
        if eval_threshold is None:
            precs, recs, thresholds = precision_recall_curve(flat_labels, flat_scores)
            f1_arr = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-8)
            if len(f1_arr) > 0:
                best_idx = int(np.argmax(f1_arr))
                opt_thresh = float(thresholds[best_idx])
            else:
                opt_thresh = 0.5
        else:
            opt_thresh = eval_threshold
            
        binary_preds = (flat_scores >= opt_thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels, binary_preds, average="binary", zero_division=0
        )
    else:
        auc_roc, pr_auc, precision, recall, f1, opt_thresh = 0.5, 0.5, 0.0, 0.0, 0.0, 0.5
        
    avg_latency = float(np.mean(inference_times))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    return {
        "trajectory_errors": disp_errors,
        "optimal_threshold": round(opt_thresh, 4),
        "anomaly_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc_roc),
            "pr_auc": float(pr_auc)
        },
        "efficiency": {
            "latency_ms": round(avg_latency, 2),
            "fps": round(fps, 1)
        }
    }

def train_and_evaluate(
    data_dir="data",
    train_recordings=[1, 2],
    val_recordings=[3],
    test_recordings=[4],
    max_samples_per_rec=300,
    epochs=10,
    batch_size=16,
    lr=1e-3,
    save_checkpoint="models/mt_stgtd_checkpoint.pt"
):
    print("="*85)
    print("  MT-STGTD Empirical Training & Benchmark Suite on highD Dataset")
    print("="*85)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    
    # 1. Prepare Datasets
    preproc = HighDDataPreprocessor(data_dir=data_dir, ego_centric=True)
    
    print("\n[Step 1/4] Loading highD Training Split...")
    train_ds = preproc.load_dataset_split(
        train_recordings, cache_name="train_highd_quick", max_samples_per_rec=max_samples_per_rec
    )
    print("\n[Step 2/4] Loading highD Validation Split...")
    val_ds = preproc.load_dataset_split(
        val_recordings, cache_name="val_highd_quick", max_samples_per_rec=max_samples_per_rec // 2
    )
    print("\n[Step 3/4] Loading highD Testing Split...")
    test_ds = preproc.load_dataset_split(
        test_recordings, cache_name="test_highd_quick", max_samples_per_rec=max_samples_per_rec // 2
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Data Splitting: Train={len(train_ds):,} scenes, Val={len(val_ds):,} scenes, Test={len(test_ds):,} scenes")
    
    # 2. Instantiate MT-STGTD Architecture
    print("\n[Step 4/4] Instantiating MT-STGTD Architecture...")
    model = MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 3. Training Loop
    best_val_ade = float("inf")
    print("\nStarting Empirical Training Pipeline...")
    print("-" * 85)
    print(f"{'Epoch':<8} | {'Total Loss':<12} | {'Traj Loss':<12} | {'Recon Loss':<12} | {'Val ADE (3s)':<14} | {'Val F1':<10}")
    print("-" * 85)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_traj_loss = 0.0
        epoch_recon_loss = 0.0
        
        for batch in train_loader:
            x = batch["x"].to(device)
            dist_mat = batch["dist_matrix"].to(device)
            y = batch["y"].to(device)
            anomaly_label = batch["anomaly_label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(x, dist_mat)
            
            # Target telemetry for reconstruction head is observed telemetry at t=T_obs-1
            target_telemetry = x[:, :, -1, :]
            
            loss, loss_dict = model.compute_loss(outputs, y, target_telemetry, anomaly_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total_loss += loss.item()
            epoch_traj_loss += loss_dict["loss_traj"]
            epoch_recon_loss += loss_dict["loss_recon"]
            
        scheduler.step()
        
        n_batches = len(train_loader)
        avg_tot = epoch_total_loss / n_batches
        avg_traj = epoch_traj_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        
        # Validation Evaluation
        val_metrics = evaluate_model(model, val_loader, device=device)
        val_ade = val_metrics["trajectory_errors"]["ade_3s"]
        val_f1 = val_metrics["anomaly_metrics"]["f1_score"]
        
        print(f"Epoch {epoch:<3}/{epochs:<2} | {avg_tot:<12.4f} | {avg_traj:<12.4f} | {avg_recon:<12.4f} | {val_ade:<14.2f}m | {val_f1:<10.3f}")
        
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            os.makedirs(os.path.dirname(save_checkpoint), exist_ok=True)
            torch.save(model.state_dict(), save_checkpoint)
            
    print("-" * 85)
    print("Training Complete. Evaluating Best Checkpoint on Unseen highD Test Split...")
    
    # 4. Final Evaluation on Test Split
    if os.path.exists(save_checkpoint):
        model.load_state_dict(torch.load(save_checkpoint, map_location=device))
        
    test_metrics = evaluate_model(model, test_loader, device=device, eval_threshold=val_metrics.get("optimal_threshold", 0.5))
    
    traj_errs = test_metrics['trajectory_errors']
    print("\n==========================================================================")
    print("  FINAL EMPIRICAL RESULTS ON UNSEEN HIGHD TEST DATA")
    print("==========================================================================")
    print("Trajectory Displacement Errors (Nominal Driving - Pure Trajectory Tracking):")
    print(f"  ADE (1.0s Horizon): {traj_errs['normal_ade_1s']:.3f} m")
    print(f"  ADE (2.0s Horizon): {traj_errs['normal_ade_2s']:.3f} m")
    print(f"  ADE (3.0s Horizon): {traj_errs['normal_ade_3s']:.3f} m")
    print(f"  FDE (3.0s Horizon): {traj_errs['normal_fde_3s']:.3f} m")
    print("\nTrajectory Displacement Errors (Overall Including Anomalies):")
    print(f"  ADE (3.0s Horizon): {traj_errs['ade_3s']:.3f} m")
    print(f"  FDE (3.0s Horizon): {traj_errs['fde_3s']:.3f} m")
    print("\nTelemetry Anomaly Detection Metrics:")
    print(f"  Precision:          {test_metrics['anomaly_metrics']['precision']:.4f}")
    print(f"  Recall:             {test_metrics['anomaly_metrics']['recall']:.4f}")
    print(f"  F1-Score:           {test_metrics['anomaly_metrics']['f1_score']:.4f}")
    print(f"  AUC-ROC:            {test_metrics['anomaly_metrics']['auc_roc']:.4f}")
    print(f"  PR-AUC:             {test_metrics['anomaly_metrics']['pr_auc']:.4f}")
    print(f"  Decision Threshold: {test_metrics['optimal_threshold']:.4f}")
    print("\nReal-Time Efficiency:")
    print(f"  Per-Frame Latency:  {test_metrics['efficiency']['latency_ms']} ms")
    print(f"  Throughput (FPS):   {test_metrics['efficiency']['fps']} FPS")
    print("==========================================================================")
    
    # Export metrics to JSON
    out_json = "experiments/highd_empirical_results.json"
    with open(out_json, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Empirical results exported to '{out_json}'.")
    return test_metrics

if __name__ == "__main__":
    train_and_evaluate(
        data_dir="data",
        train_recordings=[1, 2],
        val_recordings=[3],
        test_recordings=[4],
        max_samples_per_rec=200,
        epochs=5,
        batch_size=16
    )
