"""
Manual Verification Script: Table II (Trajectory Forecasting Benchmarks)
Paper: Real-Time Trajectory Prediction and Anomaly Detection in Large-Scale Vehicle Telemetry with Deep Learning
Target Venue: IEEE Transactions on Intelligent Transportation Systems (T-ITS)

Usage:
    python experiments/verify_table2_trajectory.py
"""

import os
import sys
import time
import json
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mt_stgtd_model import MTSTGTDModel
from models.baseline_models import LSTMAutoencoder, STGCNModel, PatchTSTModel, AASTGATModel
from experiments.run_extended_benchmarks import evaluate_model_pipeline, load_cached_dataset


def main():
    print("=" * 80)
    print("STEP 1: MANUAL VERIFICATION OF TABLE II (TRAJECTORY FORECASTING)")
    print("=" * 80)

    # 1. Load held-out test dataset
    test_path = "data/processed_cache/extended_test.npz"
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found!")
        return

    print(f"\n[1/3] Loading held-out test split: {test_path}")
    test_dataset = load_cached_dataset(test_path)
    print(f"      Total Unseen Test Scenarios: {len(test_dataset)} scenes")
    print(f"      Input observation window:    20 frames (2.0s at 10 Hz)")
    print(f"      Future prediction horizon:   30 frames (3.0s at 10 Hz)")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Checkpoint registry
    models = {
        "LSTM-Autoencoder": {
            "model": LSTMAutoencoder(),
            "ckpt": "models/checkpoints/LSTM-Autoencoder.pt",
            "paper_ade1": 1.054, "paper_ade2": 1.919, "paper_ade3": 2.834, "paper_fde3": 5.639
        },
        "ST-GCN": {
            "model": STGCNModel(),
            "ckpt": "models/checkpoints/ST-GCN.pt",
            "paper_ade1": 1.241, "paper_ade2": 1.790, "paper_ade3": 2.500, "paper_fde3": 4.912
        },
        "PatchTST": {
            "model": PatchTSTModel(),
            "ckpt": "models/checkpoints/PatchTST.pt",
            "paper_ade1": 0.422, "paper_ade2": 0.648, "paper_ade3": 0.832, "paper_fde3": 1.508
        },
        "AA-STGAT (Base Paper)": {
            "model": AASTGATModel(),
            "ckpt": "models/checkpoints/AA-STGAT_Base_Paper.pt",
            "paper_ade1": 0.422, "paper_ade2": 0.689, "paper_ade3": 0.925, "paper_fde3": 1.829
        },
        "MT-STGTD (Proposed)": {
            "model": MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20),
            "ckpt": "models/checkpoints/MT-STGTD_Proposed.pt",
            "paper_ade1": 0.282, "paper_ade2": 0.418, "paper_ade3": 0.588, "paper_fde3": 1.155
        }
    }

    # 3. Live Evaluation
    print("\n[2/3] Evaluating saved PyTorch checkpoints directly on test samples...")
    results = {}
    for name, item in models.items():
        ckpt_path = item["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"      [WARNING] Checkpoint {ckpt_path} missing!")
            continue

        model = item["model"]
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        t0 = time.time()
        res = evaluate_model_pipeline(model, test_loader, device="cpu")
        elapsed = time.time() - t0

        results[name] = res
        print(f"      * {name:<22} evaluated in {elapsed:.1f}s -> ADE(3s)={res['ade_3s']:.3f}m, FDE(3s)={res['fde_3s']:.3f}m")

    # 4. Display Table II Verification
    print("\n[3/3] COMPARISON: LIVE EVALUATION vs. MANUSCRIPT TABLE II")
    print("-" * 88)
    header = f"{'Model Architecture':<24} | {'ADE (1s)':<12} | {'ADE (2s)':<12} | {'ADE (3s)':<12} | {'FDE (3s)':<12} | {'Status'}"
    print(header)
    print("-" * 88)

    all_matched = True
    for name, item in models.items():
        if name not in results:
            continue
        res = results[name]
        calc_ade1 = res["ade_1s"]
        calc_ade2 = res["ade_2s"]
        calc_ade3 = res["ade_3s"]
        calc_fde3 = res["fde_3s"]

        paper_ade1 = item["paper_ade1"]
        paper_ade2 = item["paper_ade2"]
        paper_ade3 = item["paper_ade3"]
        paper_fde3 = item["paper_fde3"]

        match_ade1 = abs(calc_ade1 - paper_ade1) < 0.005
        match_ade2 = abs(calc_ade2 - paper_ade2) < 0.005
        match_ade3 = abs(calc_ade3 - paper_ade3) < 0.005
        match_fde3 = abs(calc_fde3 - paper_fde3) < 0.005

        match = match_ade1 and match_ade2 and match_ade3 and match_fde3
        status_str = "EXACT MATCH [PASS]" if match else "MISMATCH [FAIL]"
        if not match:
            all_matched = False

        row = (
            f"{name:<24} | "
            f"{calc_ade1:.3f}m ({paper_ade1:.3f}) | "
            f"{calc_ade2:.3f}m ({paper_ade2:.3f}) | "
            f"{calc_ade3:.3f}m ({paper_ade3:.3f}) | "
            f"{calc_fde3:.3f}m ({paper_fde3:.3f}) | "
            f"{status_str}"
        )
        print(row)

    print("-" * 88)
    print("Format in columns above: Computed_Metric (Manuscript_Metric)")

    # 5. Verification of Percentage Reduction Claims
    if "MT-STGTD (Proposed)" in results and "AA-STGAT (Base Paper)" in results:
        prop_ade3 = results["MT-STGTD (Proposed)"]["ade_3s"]
        base_ade3 = results["AA-STGAT (Base Paper)"]["ade_3s"]
        prop_fde3 = results["MT-STGTD (Proposed)"]["fde_3s"]
        base_fde3 = results["AA-STGAT (Base Paper)"]["fde_3s"]

        ade3_red = ((base_ade3 - prop_ade3) / base_ade3) * 100.0
        fde3_red = ((base_fde3 - prop_fde3) / base_fde3) * 100.0

        print("\n=== VERIFICATION OF PAPER CLAIMS (Section V-A) ===")
        print(f"1. 3.0s ADE Reduction vs. Base Paper (AA-STGAT):")
        print(f"   ({base_ade3:.3f}m - {prop_ade3:.3f}m) / {base_ade3:.3f}m = {ade3_red:.2f}%")
        print(f"   Reported in Paper: 36.4%  ->  Status: {'VERIFIED [PASS]' if abs(ade3_red - 36.4) < 0.5 else 'FAIL'}")

        print(f"\n2. 3.0s FDE Reduction vs. Base Paper (AA-STGAT):")
        print(f"   ({base_fde3:.3f}m - {prop_fde3:.3f}m) / {base_fde3:.3f}m = {fde3_red:.2f}%")
        print(f"   Reported in Paper: 36.9%  ->  Status: {'VERIFIED [PASS]' if abs(fde3_red - 36.9) < 0.5 else 'FAIL'}")

    print("=" * 80)
    if all_matched:
        print("RESULT: ALL TABLE II METRICS SUCCESSFULLY AND MANUALLY VERIFIED!")
    else:
        print("RESULT: SOME METRICS DO NOT MATCH!")
    print("=" * 80)


if __name__ == "__main__":
    main()
