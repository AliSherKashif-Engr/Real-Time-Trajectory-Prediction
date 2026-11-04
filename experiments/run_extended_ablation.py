"""
Extended Component-Wise Architectural Ablation Study on Authentic highD Dataset
Evaluates variants M1 through M5 on extended_train.npz and extended_test.npz,
and pairs them with M6 (Full MT-STGTD) to provide 100% empirical evidence for Table IV.
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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_ablation_study import (
    M1_Baseline, M2_7D_Telemetry, M3_DGAT, M4_PatchSTGT, M5_KinematicResidual
)
from experiments.run_extended_benchmarks import load_cached_dataset
from models.mt_stgtd_model import MTSTGTDModel

def evaluate_ablation_variant(model, test_loader, device="cpu"):
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
    
    return {
        "ade_1s": round(ade_1s, 3),
        "ade_3s": round(ade_3s, 3),
        "fde_3s": round(fde_3s, 3),
        "f1_score": round(best_f1, 4),
        "auc_roc": round(auc_roc, 4),
        "latency_ms": round(avg_lat, 2),
        "params": f"{total_p/1e6:.2f}M"
    }

def run_extended_ablation():
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print("=" * 90)
    print("  Paper 01: Extended Architectural Ablation Study (M1 through M6)")
    print("=" * 90)
    
    train_cache = "data/processed_cache/extended_train.npz"
    test_cache = "data/processed_cache/extended_test.npz"
    
    train_ds = load_cached_dataset(train_cache)
    test_ds = load_cached_dataset(test_cache)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    variants = [
        ("M1: Baseline ST-GNN (2D)", M1_Baseline(), 6),
        ("M2: + 7D Telemetry Vector", M2_7D_Telemetry(), 6),
        ("M3: + Dynamic Spatial GAT", M3_DGAT(), 6),
        ("M4: + Patchified Encoder", M4_PatchSTGT(), 6),
        ("M5: + Kinematic Residual Prior", M5_KinematicResidual(), 6),
    ]
    
    ablation_results = {}
    
    for name, model, epochs in variants:
        print(f"\nEvaluating Ablation Variant: [{name}] for {epochs} epochs...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.to(device)
        
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                dist_mat = batch["dist_matrix"].to(device)
                y = batch["y"].to(device)
                labels = batch["anomaly_label"].to(device)
                
                optimizer.zero_grad()
                out = model(x, dist_mat)
                target_telemetry = x[:, :, -1, :]
                
                # Weighted BCE for fair anomaly supervision
                loss_traj = F.mse_loss(out["pred_traj"], y)
                cls_w = torch.where(labels == 1.0, torch.tensor(5.0), torch.tensor(1.0))
                loss_cls = F.binary_cross_entropy(out["anomaly_score"], labels, weight=cls_w)
                loss = loss_traj + 0.5 * loss_cls
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
        res = evaluate_ablation_variant(model, test_loader, device=device)
        ablation_results[name] = res
        print(f"  --> {name}: ADE(1s)={res['ade_1s']}m, ADE(3s)={res['ade_3s']}m, F1={res['f1_score']}, AUC={res['auc_roc']}")
        
    # Load M6 from extended_benchmark_results.json
    bench_results_path = "experiments/extended_benchmark_results.json"
    with open(bench_results_path, "r") as f:
        bench_data = json.load(f)
    m6_data = bench_data["MT-STGTD (Proposed)"]
    
    ablation_results["M6: Full MT-STGTD (+ Energy Head & Uncertainty Loss)"] = {
        "ade_1s": m6_data["ade_1s"],
        "ade_3s": m6_data["ade_3s"],
        "fde_3s": m6_data["fde_3s"],
        "f1_score": m6_data["f1_score"],
        "auc_roc": m6_data["auc_roc"],
        "latency_ms": m6_data["latency_ms"],
        "params": m6_data["params"]
    }
    
    out_file = "experiments/ablation_study_results.json"
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=4)
    print(f"\nSaved updated ablation study results to '{out_file}'.")
    
    print("\n" + "=" * 105)
    print("Table IV: Component-Wise Ablation Study Results on highD Benchmark")
    print("=" * 105)
    print(f"{'Ablation Variant':<45} | {'ADE (1s)':<8} | {'ADE (3s)':<8} | {'FDE (3s)':<8} | {'F1-Score':<8} | {'AUC-ROC':<8} | {'Latency':<8}")
    print("=" * 105)
    for v_name, m in ablation_results.items():
        print(f"{v_name:<45} | {m['ade_1s']:<8.3f} | {m['ade_3s']:<8.3f} | {m['fde_3s']:<8.3f} | {m['f1_score']:<8.4f} | {m['auc_roc']:<8.4f} | {m['latency_ms']:<6.2f}ms")
    print("=" * 105)
    return ablation_results

if __name__ == "__main__":
    run_extended_ablation()
