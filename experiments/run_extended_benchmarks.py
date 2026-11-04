"""
Extended Empirical Benchmark Runner on Authentic highD Dataset
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)

Executes extended training and evaluation across all 5 architectures:
1. LSTM-Autoencoder (2D Coordinates)
2. ST-GCN (3D Kinematics)
3. PatchTST (4D Kinematics)
4. AA-STGAT (Base Paper Benchmark - 4D Kinematics)
5. MT-STGTD (Proposed SOTA Architecture - 7D Multi-Modal Telemetry)

Saves 100% authentic metrics, ROC arrays, and qualitative trajectories to disk.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    roc_curve, precision_recall_curve
)

# Paths and imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mt_stgtd_model import MTSTGTDModel
from models.baseline_models import LSTMAutoencoder, STGCNModel, PatchTSTModel, AASTGATModel
from experiments.highd_dataset_loader import HighDDataset

def load_cached_dataset(cache_path):
    """Loads pre-cached NPZ split into HighDDataset."""
    data = np.load(cache_path)
    return HighDDataset(
        x_data=data["x"],
        y_data=data["y"],
        dist_matrices=data["dist"],
        anomaly_labels=data["label"],
        masks=data["mask"]
    )

def evaluate_model_pipeline(model, data_loader, device="cpu", eval_threshold=None, return_curves=False):
    model.eval()
    all_preds, all_targets, all_masks, all_labels = [], [], [], []
    all_scores = []
    latencies = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            dist_mat = batch["dist_matrix"].to(device)
            y = batch["y"].to(device)
            labels = batch["anomaly_label"].to(device)
            mask = batch["mask"].to(device)
            
            t0 = time.perf_counter()
            out = model(x, dist_mat)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms / len(x))
            
            all_preds.append(out["pred_traj"].cpu())
            all_targets.append(y.cpu())
            all_masks.append(mask.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(out["anomaly_score"][:, 0, 0].cpu().numpy())
            
    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    flat_scores = np.concatenate(all_scores)
    flat_labels = cat_labels[:, 0, 0].numpy()
    
    # Trajectory Errors on Nominal Driving (ground truth trajectory tracking)
    diff = cat_preds[:, 0, :, :] - cat_targets[:, 0, :, :]
    euc_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1)) # [N_samples, 30]
    normal_mask = (cat_labels[:, 0, 0] == 0)
    
    if normal_mask.sum() > 0:
        norm_dist = euc_dist[normal_mask]
        ade_1s = float(norm_dist[:, :10].mean().item())
        ade_2s = float(norm_dist[:, :20].mean().item())
        ade_3s = float(norm_dist[:, :30].mean().item())
        fde_3s = float(norm_dist[:, 29].mean().item())
    else:
        ade_1s = float(euc_dist[:, :10].mean().item())
        ade_2s = float(euc_dist[:, :20].mean().item())
        ade_3s = float(euc_dist[:, :30].mean().item())
        fde_3s = float(euc_dist[:, 29].mean().item())
        
    # Anomaly Detection Metrics
    curves_data = {}
    if len(np.unique(flat_labels)) > 1:
        auc_roc = float(roc_auc_score(flat_labels, flat_scores))
        pr_auc = float(average_precision_score(flat_labels, flat_scores))
        
        precs, recs, threshs = precision_recall_curve(flat_labels, flat_scores)
        f1_arr = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-8)
        best_idx = int(np.argmax(f1_arr)) if len(f1_arr) > 0 else 0
        opt_thresh = float(threshs[best_idx]) if len(threshs) > best_idx else 0.5
        
        if eval_threshold is not None:
            active_thresh = eval_threshold
        else:
            active_thresh = opt_thresh
            
        bin_preds = (flat_scores >= active_thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(flat_labels, bin_preds, average="binary", zero_division=0)
        
        if return_curves:
            fpr, tpr, roc_threshs = roc_curve(flat_labels, flat_scores)
            curves_data["fpr"] = fpr
            curves_data["tpr"] = tpr
            curves_data["precs"] = precs
            curves_data["recs"] = recs
            curves_data["flat_scores"] = flat_scores
            curves_data["flat_labels"] = flat_labels
    else:
        auc_roc, pr_auc, p, r, f1, opt_thresh = 0.5, 0.5, 0.0, 0.0, 0.0, 0.5
        
    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    result = {
        "ade_1s": round(ade_1s, 3),
        "ade_2s": round(ade_2s, 3),
        "ade_3s": round(ade_3s, 3),
        "fde_3s": round(fde_3s, 3),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc_roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "opt_thresh": round(opt_thresh, 4),
        "latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1)
    }
    
    if return_curves:
        return result, curves_data, cat_preds, cat_targets
    return result

def train_and_eval_model(model_name, model, train_loader, val_loader, test_loader, epochs=15, lr=1e-3, device="cpu", save_dir="models/checkpoints"):
    print(f"\n==========================================================================")
    print(f"  Training Architecture: [{model_name}] for {epochs} Epochs")
    print(f"==========================================================================")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.pt")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    
    best_val_score = float("inf")
    best_thresh = 0.5
    
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val ADE (3s)':<14} | {'Val F1':<10} | {'Val AUC':<10}")
    print("-" * 65)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x = batch["x"].to(device)
            dist_mat = batch["dist_matrix"].to(device)
            y = batch["y"].to(device)
            anomaly_label = batch["anomaly_label"].to(device)
            
            optimizer.zero_grad()
            out = model(x, dist_mat)
            
            if hasattr(model, "compute_loss") and model.__class__.__name__ == "MTSTGTDModel":
                target_telemetry = x[:, :, -1, :]
                loss, _ = model.compute_loss(out, y, target_telemetry, anomaly_label)
            else:
                loss, _ = model.compute_loss(out, y, anomaly_label)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        avg_train_loss = train_loss / max(1, n_batches)
        
        val_res = evaluate_model_pipeline(model, val_loader, device=device)
        print(f"Epoch {epoch:<3}/{epochs:<2} | {avg_train_loss:<12.4f} | {val_res['ade_3s']:<14.3f}m | {val_res['f1_score']:<10.4f} | {val_res['auc_roc']:<10.4f}")
        
        # Save best model checkpoint based on ADE & Anomaly balanced criterion
        score = val_res["ade_3s"] - 0.5 * val_res["auc_roc"]
        if score < best_val_score:
            best_val_score = score
            best_thresh = val_res["opt_thresh"]
            torch.save(model.state_dict(), checkpoint_path)
            
    # Load best checkpoint for test evaluation
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    test_res, curves, preds, targets = evaluate_model_pipeline(
        model, test_loader, device=device, eval_threshold=best_thresh, return_curves=True
    )
    total_params = sum(p.numel() for p in model.parameters())
    test_res["params"] = f"{total_params/1e6:.2f}M"
    
    print(f"\n--> Test Results for [{model_name}]:")
    print(f"    ADE(1s)={test_res['ade_1s']}m, ADE(2s)={test_res['ade_2s']}m, ADE(3s)={test_res['ade_3s']}m, FDE(3s)={test_res['fde_3s']}m")
    print(f"    F1={test_res['f1_score']}, AUC-ROC={test_res['auc_roc']}, Latency={test_res['latency_ms']}ms")
    
    return test_res, curves, preds, targets

def run_extended_experiments():
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print(f"Hardware Compute Platform: {device} (8 CPU threads)")
    
    cache_dir = "data/processed_cache"
    train_cache = os.path.join(cache_dir, "extended_train.npz")
    val_cache = os.path.join(cache_dir, "extended_val.npz")
    test_cache = os.path.join(cache_dir, "extended_test.npz")
    
    print(f"Loading cached dataset splits from '{cache_dir}'...")
    train_ds = load_cached_dataset(train_cache)
    val_ds = load_cached_dataset(val_cache)
    test_ds = load_cached_dataset(test_cache)
    
    print(f"Loaded: Train={len(train_ds):,} samples, Val={len(val_ds):,} samples, Test={len(test_ds):,} samples.")
    
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    architectures = [
        ("LSTM-Autoencoder", LSTMAutoencoder()),
        ("ST-GCN", STGCNModel()),
        ("PatchTST", PatchTSTModel()),
        ("AA-STGAT (Base Paper)", AASTGATModel()),
        ("MT-STGTD (Proposed)", MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20))
    ]
    
    all_metrics = {}
    all_curves = {}
    qualitative_data = {}
    
    for name, model in architectures:
        res, curves, preds, targets = train_and_eval_model(
            name, model, train_loader, val_loader, test_loader, epochs=12, lr=1e-3, device=device
        )
        all_metrics[name] = res
        all_curves[name] = {
            "fpr": curves["fpr"].tolist() if "fpr" in curves else [],
            "tpr": curves["tpr"].tolist() if "tpr" in curves else [],
            "precs": curves["precs"].tolist() if "precs" in curves else [],
            "recs": curves["recs"].tolist() if "recs" in curves else []
        }
        
        if name == "MT-STGTD (Proposed)":
            qualitative_data["preds"] = preds[:10].numpy()
            qualitative_data["targets"] = targets[:10].numpy()
            
    # Export metrics JSON
    metrics_path = "experiments/extended_benchmark_results.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nSaved benchmark metrics to '{metrics_path}'.")
    
    # Export curves and qualitative data
    curves_path = "experiments/extended_curves_data.json"
    with open(curves_path, "w") as f:
        json.dump(all_curves, f)
    print(f"Saved ROC curves data to '{curves_path}'.")
    
    # Save qualitative test trajectories
    np.savez_compressed("experiments/extended_qualitative_samples.npz", **qualitative_data)
    print("Saved qualitative samples to 'experiments/extended_qualitative_samples.npz'.")
    
    print("\n" + "=" * 115)
    print(f"{'Model Architecture':<24} | {'ADE (1s)':<8} | {'ADE (2s)':<8} | {'ADE (3s)':<8} | {'FDE (3s)':<8} | {'F1-Score':<8} | {'AUC-ROC':<8} | {'Latency':<8}")
    print("=" * 115)
    for model_name, m in all_metrics.items():
        print(f"{model_name:<24} | {m['ade_1s']:<8.3f} | {m['ade_2s']:<8.3f} | {m['ade_3s']:<8.3f} | {m['fde_3s']:<8.3f} | {m['f1_score']:<8.4f} | {m['auc_roc']:<8.4f} | {m['latency_ms']:<6.2f}ms")
    print("=" * 115)
    
    return all_metrics

if __name__ == "__main__":
    run_extended_experiments()
