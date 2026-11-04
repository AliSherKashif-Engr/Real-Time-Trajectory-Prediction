"""
Computational Efficiency, FLOPs & Real-Time Latency Profiler for Paper 01
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)

Profiles computational resource footprints and inference latency across:
1. LSTM-Autoencoder
2. ST-GCN
3. PatchTST
4. AA-STGAT (Base Paper)
5. MT-STGTD (Proposed Architecture)

Measures:
  - Total Parameter Count (Millions)
  - Theoretical Computational Complexity (FLOPs / GFLOPs)
  - Edge Inference Latency (Batch Size = 1, single vehicle stream)
  - Cloud Fleet Ingestion Latency (Batch Size = 32)
  - Throughput (Frames Per Second / FPS)
  - 10ms Real-Time SLA Compliance Flag
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.baseline_models import LSTMAutoencoder, STGCNModel, PatchTSTModel, AASTGATModel
from models.mt_stgtd_model import MTSTGTDModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, x, dist_matrix):
    """
    Estimates floating point operations (FLOPs) during a single forward pass.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(x, dist_matrix)
        total_flops = flop_counter.get_total_flops()
        return total_flops / 1e9 # GFLOPs
    except Exception:
        # Theoretical estimation fallback
        total_params = count_parameters(model)
        # Approximate 2 FLOPs per parameter per sequence step
        T_steps = x.shape[2]
        approx_flops = 2.0 * total_params * T_steps
        return approx_flops / 1e9

def profile_latency(model, batch_size=1, n_vehicles=6, obs_steps=20, n_runs=100, device="cpu"):
    """
    Measures warm-up and multi-iteration execution latency.
    """
    model.eval()
    model.to(device)
    
    x = torch.randn(batch_size, n_vehicles, obs_steps, 7, device=device)
    dist = torch.rand(batch_size, obs_steps, n_vehicles, n_vehicles, device=device) * 30.0
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(15):
            _ = model(x, dist)
            
    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x, dist)
            t_ms = (time.perf_counter() - t0) * 1000.0 # ms
            timings.append(t_ms)
            
    mean_batch_lat = float(sum(timings) / len(timings))
    per_frame_lat = mean_batch_lat / batch_size
    fps = 1000.0 / per_frame_lat if per_frame_lat > 0 else 0.0
    
    return round(per_frame_lat, 2), round(fps, 1)

def run_hardware_benchmarking():
    print("="*95)
    print("  Paper 01: Hardware Resource Footprint & Real-Time Latency Benchmark")
    print("  Comparing MT-STGTD against Baseline SOTA Architectures")
    print("="*95)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling Compute Device: {device}")
    
    models = {
        "LSTM-Autoencoder": LSTMAutoencoder(),
        "ST-GCN": STGCNModel(),
        "PatchTST": PatchTSTModel(),
        "AA-STGAT (Base Paper)": AASTGATModel(),
        "MT-STGTD (Proposed)": MTSTGTDModel(in_features=7, d_model=128, pred_steps=30, obs_steps=20)
    }
    
    results = {}
    dummy_x = torch.randn(1, 6, 20, 7)
    dummy_dist = torch.rand(1, 20, 6, 6)
    
    print("\n" + "="*115)
    print(f"{'Model Architecture':<24} | {'Params (M)':<12} | {'FLOPs (G)':<10} | {'Edge Lat (B=1)':<15} | {'Cloud Lat (B=32)':<16} | {'FPS':<10} | {'SLA (<10ms)':<10}")
    print("="*115)
    
    for name, model in models.items():
        params = count_parameters(model)
        params_m = round(params / 1e6, 2)
        flops_g = round(estimate_flops(model, dummy_x, dummy_dist), 2)
        
        # Edge Latency (B=1)
        edge_lat, edge_fps = profile_latency(model, batch_size=1, n_runs=100, device=device)
        # Cloud Latency (B=32)
        cloud_lat, cloud_fps = profile_latency(model, batch_size=32, n_runs=50, device=device)
        
        sla_pass = "PASSED" if edge_lat < 10.0 else "FAILED"
        
        results[name] = {
            "parameters_m": params_m,
            "flops_g": flops_g,
            "edge_latency_ms": edge_lat,
            "edge_fps": edge_fps,
            "cloud_latency_per_sample_ms": cloud_lat,
            "cloud_fps": cloud_fps,
            "sla_compliance": sla_pass
        }
        
        print(f"{name:<24} | {params_m:<12.2f} | {flops_g:<10.2f} | {edge_lat:<13.2f}ms | {cloud_lat:<14.2f}ms | {edge_fps:<10.1f} | {sla_pass:<10}")
        
    print("="*115)
    
    out_file = "experiments/computational_profile_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nComputational profile successfully exported to '{out_file}'.")
    return results

if __name__ == "__main__":
    run_hardware_benchmarking()
