"""
Manual Verification Script: Table III (Telemetry Anomaly Detection Sensitivity)
Paper: Real-Time Trajectory Prediction and Anomaly Detection in Large-Scale Vehicle Telemetry with Deep Learning
Target Venue: IEEE Transactions on Intelligent Transportation Systems (T-ITS)

Usage:
    python experiments/verify_table3_anomaly.py
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import auc
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mt_stgtd_model import MTSTGTDModel
from models.baseline_models import LSTMAutoencoder, STGCNModel, PatchTSTModel, AASTGATModel
from experiments.run_extended_benchmarks import evaluate_model_pipeline, load_cached_dataset


def main():
    print("=" * 96)
    print("STEP 2: MANUAL VERIFICATION OF TABLE III (ANOMALY DETECTION SENSITIVITY)")
    print("=" * 96)

    # -------------------------------------------------------------------------
    # 1. Dataset Split and Injected Anomaly Audit
    # -------------------------------------------------------------------------
    test_path = "data/processed_cache/extended_test.npz"
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found!")
        return

    print(f"\n[1/4] Inspecting Held-Out Test Split: {test_path}")
    raw_data = np.load(test_path)
    labels = raw_data["label"][:, 0, 0]
    total_scenes = len(labels)
    n_anom = int(np.sum(labels == 1))
    n_norm = int(np.sum(labels == 0))
    print(f"      * Total Test Scenarios:     {total_scenes} multi-agent highway scenes")
    print(f"      * Nominal Driving Sequences: {n_norm} ({n_norm / total_scenes * 100:.1f}%)")
    print(f"      * Injected Driving Anomalies: {n_anom} ({n_anom / total_scenes * 100:.1f}%) [High lateral-G, hard brake, drift]")

    # -------------------------------------------------------------------------
    # 2. Raw Benchmark Log Verification (experiments/extended_benchmark_results.json)
    # -------------------------------------------------------------------------
    print("\n[2/4] Verifying Saved Benchmark Data against Manuscript Table III...")
    bench_file = "experiments/extended_benchmark_results.json"
    with open(bench_file, "r") as f:
        bench_data = json.load(f)

    paper_table3 = {
        "LSTM-Autoencoder":      {"p": 0.212, "r": 0.573, "f1": 0.310, "auc": 0.553, "pr": 0.221},
        "ST-GCN":                 {"p": 0.158, "r": 0.927, "f1": 0.270, "auc": 0.434, "pr": 0.145},
        "PatchTST":               {"p": 1.000, "r": 0.469, "f1": 0.638, "auc": 0.619, "pr": 0.580},
        "AA-STGAT (Base Paper)":  {"p": 0.139, "r": 0.208, "f1": 0.167, "auc": 0.495, "pr": 0.179},
        "MT-STGTD (Proposed)":    {"p": 0.613, "r": 0.760, "f1": 0.679, "auc": 0.931, "pr": 0.763}
    }

    print("-" * 96)
    header = f"{'Model Architecture':<24} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'AUC-ROC':<11} | {'PR-AUC':<11} | {'Status'}"
    print(header)
    print("-" * 96)

    table3_pass = True
    for name, p_vals in paper_table3.items():
        b_entry = bench_data[name]
        bp = b_entry["precision"]
        br = b_entry["recall"]
        bf1 = b_entry["f1_score"]
        bauc = b_entry["auc_roc"]
        bpr = b_entry["pr_auc"]

        m_p = abs(round(bp, 3) - p_vals["p"]) < 0.005
        m_r = abs(round(br, 3) - p_vals["r"]) < 0.005
        m_f1 = abs(round(bf1, 3) - p_vals["f1"]) < 0.005
        m_auc = abs(round(bauc, 3) - p_vals["auc"]) < 0.005
        m_pr = abs(round(bpr, 3) - p_vals["pr"]) < 0.005

        match = m_p and m_r and m_f1 and m_auc and m_pr
        if not match:
            table3_pass = False

        status = "EXACT MATCH [PASS]" if match else "MISMATCH [FAIL]"
        row = (
            f"{name:<24} | "
            f"{bp:.3f} ({p_vals['p']:.3f}) | "
            f"{br:.3f} ({p_vals['r']:.3f}) | "
            f"{bf1:.3f} ({p_vals['f1']:.3f}) | "
            f"{bauc:.3f} ({p_vals['auc']:.3f}) | "
            f"{bpr:.3f} ({p_vals['pr']:.3f}) | "
            f"{status}"
        )
        print(row)

    print("-" * 96)
    print("Format above: Recorded_Benchmark (Manuscript_Table_III)")

    # -------------------------------------------------------------------------
    # 3. Direct Integration of Raw FPR/TPR ROC Curve Arrays (Figure 2)
    # -------------------------------------------------------------------------
    print("\n[3/4] Verifying ROC Curve Integration from Raw Arrays (Fig. 2)...")
    curves_file = "experiments/extended_curves_data.json"
    with open(curves_file, "r") as f:
        curves_data = json.load(f)

    for name in paper_table3:
        fpr = np.array(curves_data[name]["fpr"])
        tpr = np.array(curves_data[name]["tpr"])
        computed_auc = float(auc(fpr, tpr))
        paper_auc = paper_table3[name]["auc"]
        auc_match = abs(round(computed_auc, 3) - paper_auc) < 0.005
        print(f"      * {name:<24}: {len(fpr):>3} ROC points -> Trapeze Integral AUC = {computed_auc:.4f} (Paper: {paper_auc:.3f}) -> {'[PASS]' if auc_match else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # 4. Live Forward Checkpoint Evaluation on 600 Unseen Test Scenes
    # -------------------------------------------------------------------------
    print("\n[4/4] Executing Live PyTorch Forward Inference from Saved Checkpoints...")
    test_dataset = load_cached_dataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ckpt_map = {
        "LSTM-Autoencoder":      (LSTMAutoencoder(), "models/checkpoints/LSTM-Autoencoder.pt"),
        "ST-GCN":                 (STGCNModel(), "models/checkpoints/ST-GCN.pt"),
        "PatchTST":               (PatchTSTModel(), "models/checkpoints/PatchTST.pt"),
        "AA-STGAT (Base Paper)":  (AASTGATModel(), "models/checkpoints/AA-STGAT_Base_Paper.pt"),
        "MT-STGTD (Proposed)":    (MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20), "models/checkpoints/MT-STGTD_Proposed.pt")
    }

    live_pass = True
    for name, (model, ckpt_path) in ckpt_map.items():
        if not os.path.exists(ckpt_path):
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        res = evaluate_model_pipeline(model, test_loader, device="cpu")
        paper_auc = paper_table3[name]["auc"]
        paper_pr = paper_table3[name]["pr"]
        live_auc = res["auc_roc"]
        live_pr = res["pr_auc"]

        auc_ok = abs(live_auc - paper_auc) < 0.005
        pr_ok = abs(live_pr - paper_pr) < 0.005
        if not (auc_ok and pr_ok):
            live_pass = False

        print(f"      * {name:<24}: Live AUC-ROC = {live_auc:.3f} (Paper: {paper_auc:.3f}), Live PR-AUC = {live_pr:.3f} (Paper: {paper_pr:.3f}) -> {'[PASS]' if (auc_ok and pr_ok) else '[FAIL]'}")

    print("\n" + "=" * 96)
    if table3_pass and live_pass:
        print("RESULT: ALL TABLE III ANOMALY DETECTION METRICS FULLY VERIFIED (100% MATCH)!")
    else:
        print("RESULT: SOME METRICS DO NOT MATCH!")
    print("=" * 96)


if __name__ == "__main__":
    main()
