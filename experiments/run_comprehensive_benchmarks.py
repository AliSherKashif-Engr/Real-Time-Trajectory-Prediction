"""
Comprehensive Empirical Benchmark Runner on Authentic highD Dataset
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)

Executes comparative benchmarking of MT-STGTD against all 4 baseline architectures:
1. LSTM-Autoencoder (2D Coordinates)
2. ST-GCN (3D Kinematics)
3. PatchTST (4D Kinematics)
4. AA-STGAT (Base Paper Benchmark - 4D Kinematics)
5. MT-STGTD (Proposed SOTA Architecture - 7D Multi-Modal Telemetry)

Evaluates:
  - Trajectory Forecasting: ADE (1.0s, 2.0s, 3.0s), FDE (3.0s)
  - Anomaly Detection: Precision, Recall, F1-Score, AUC-ROC, PR-AUC
  - Latency & Computational Efficiency: Latency (ms), Throughput (FPS), Parameter count
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, precision_recall_curve

# Paths and imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mt_stgtd_model import MTSTGTDModel
from models.baseline_models import LSTMAutoencoder, STGCNModel, PatchTSTModel, AASTGATModel
from experiments.highd_dataset_loader import HighDDataPreprocessor

def evaluate_model_pipeline(model, data_loader, device="cpu", eval_threshold=None):
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
    
    # Trajectory Errors on Nominal Driving (Ground truth trajectory tracking)
    diff = cat_preds[:, 0, :, :] - cat_targets[:, 0, :, :]
    euc_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1)) # [B, 30]
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
    if len(np.unique(flat_labels)) > 1:
        auc_roc = float(roc_auc_score(flat_labels, flat_scores))
        pr_auc = float(average_precision_score(flat_labels, flat_scores))
        
        if eval_threshold is None:
            precs, recs, threshs = precision_recall_curve(flat_labels, flat_scores)
            f1_arr = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-8)
            best_idx = int(np.argmax(f1_arr)) if len(f1_arr) > 0 else 0
            opt_thresh = float(threshs[best_idx]) if len(threshs) > best_idx else 0.5
        else:
            opt_thresh = eval_threshold
            
        bin_preds = (flat_scores >= opt_thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(flat_labels, bin_preds, average="binary", zero_division=0)
    else:
        auc_roc, pr_auc, p, r, f1, opt_thresh = 0.5, 0.5, 0.0, 0.0, 0.0, 0.5
        
    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    return {
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

def train_and_eval_architecture(model_name, model, train_loader, val_loader, test_loader, epochs=6, lr=1e-3, device="cpu"):
    print(f"\nTraining and Benchmarking: [{model_name}]...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    best_val_score = float("inf")
    best_thresh = 0.5
    
    for epoch in range(1, epochs + 1):
        model.train()
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
            
        val_res = evaluate_model_pipeline(model, val_loader, device=device)
        if val_res["ade_3s"] < best_val_score:
            best_val_score = val_res["ade_3s"]
            best_thresh = val_res["opt_thresh"]
            
    # Final evaluation on unseen test split
    test_res = evaluate_model_pipeline(model, test_loader, device=device, eval_threshold=best_thresh)
    total_params = sum(p.numel() for p in model.parameters())
    test_res["params"] = f"{total_params/1e6:.2f}M"
    return test_res

def run_all_benchmarks():
    print("="*95)
    print("  Paper 01: Complete Empirical Benchmarking Suite on highD Dataset")
    print("  Comparing MT-STGTD against 4 Baseline SOTA Architectures")
    print("="*95)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Platform: {device}")
    
    # Load dataset splits from cache
    preproc = HighDDataPreprocessor(data_dir="data", ego_centric=True)
    train_ds = preproc.load_dataset_split([1, 2], cache_name="train_highd_quick")
    val_ds = preproc.load_dataset_split([3], cache_name="val_highd_quick")
    test_ds = preproc.load_dataset_split([4], cache_name="test_highd_quick")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    architectures = [
        ("LSTM-Autoencoder", LSTMAutoencoder()),
        ("ST-GCN", STGCNModel()),
        ("PatchTST", PatchTSTModel()),
        ("AA-STGAT (Base Paper)", AASTGATModel()),
        ("MT-STGTD (Proposed)", MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20))
    ]
    
    benchmark_results = {}
    
    for name, model in architectures:
        res = train_and_eval_architecture(
            name, model, train_loader, val_loader, test_loader, epochs=6, lr=1e-3, device=device
        )
        benchmark_results[name] = res
        print(f"  --> {name}: ADE(3s)={res['ade_3s']}m, FDE(3s)={res['fde_3s']}m, F1={res['f1_score']}, AUC={res['auc_roc']}, Latency={res['latency_ms']}ms")
        
    print("\n" + "="*115)
    print(f"{'Model Architecture':<24} | {'ADE (1s)':<8} | {'ADE (2s)':<8} | {'ADE (3s)':<8} | {'FDE (3s)':<8} | {'F1-Score':<8} | {'AUC-ROC':<8} | {'Latency':<8}")
    print("="*115)
    for model_name, metrics in benchmark_results.items():
        print(f"{model_name:<24} | {metrics['ade_1s']:<8.3f} | {metrics['ade_2s']:<8.3f} | {metrics['ade_3s']:<8.3f} | {metrics['fde_3s']:<8.3f} | {metrics['f1_score']:<8.4f} | {metrics['auc_roc']:<8.4f} | {metrics['latency_ms']:<6.2f}ms")
    print("="*115)
    
    out_file = "experiments/comprehensive_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"\nAll benchmark results successfully exported to '{out_file}'.")
    return benchmark_results

if __name__ == "__main__":
    run_all_benchmarks()
