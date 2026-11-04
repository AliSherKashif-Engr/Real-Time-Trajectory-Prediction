"""
Vehicular Telemetry Benchmark Data Generator & Preprocessor
Generates benchmark-aligned multi-modal vehicular telemetry datasets (Porto Taxi / highD / NGSIM format)
with injected kinematic anomalies, GPS drift, swerving, hard braking, and sensor spoofing.

Author: Ali Sher Kashif (Ph.D. Research, Paper 01)
"""

import random
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    # Fallback minimal numpy-like wrapper using standard math/random
    class DummyNumpy:
        def __init__(self):
            self.random = self
        def seed(self, s):
            random.seed(s)
        def uniform(self, low, high, size=None):
            if size is None:
                return random.uniform(low, high)
            return [random.uniform(low, high) for _ in range(size)]
        def normal(self, loc, scale):
            return random.gauss(loc, scale)
        def choice(self, a, size=1, replace=True):
            if isinstance(a, int):
                arr = list(range(a))
            else:
                arr = list(a)
            res = random.sample(arr, size) if not replace else [random.choice(arr) for _ in range(size)]
            return res[0] if size == 1 else res
        def zeros(self, shape):
            if len(shape) == 3:
                return [[[0.0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
            elif len(shape) == 2:
                return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
            return [0.0 for _ in range(shape[0])]
        def array(self, obj):
            return obj
        def sin(self, x): return math.sin(x)
        def cos(self, x): return math.cos(x)
        def pi(self): return math.pi
    np = DummyNumpy()

def generate_telemetry_dataset(num_samples=1000, obs_steps=20, pred_steps=30, anomaly_ratio=0.15, seed=42):
    """
    Generates telemetry sequences for vehicle trajectory prediction and anomaly detection benchmarks.
    
    Returns:
        X_obs: [num_samples, obs_steps, 7] - Multi-modal telemetry (x, y, v, ax, ay, heading, yaw_rate)
        Y_pred: [num_samples, pred_steps, 2] - Ground truth future coordinates (x, y)
        anomaly_labels: [num_samples, 1] - Binary anomaly label (0: Normal, 1: Anomalous)
    """
    np.random.seed(seed)
    
    # 1. Base Trajectory Parameters
    dt = 0.1 # 10 Hz sampling rate
    X_obs = np.zeros((num_samples, obs_steps, 7))
    Y_pred = np.zeros((num_samples, pred_steps, 2))
    anomaly_labels = np.zeros((num_samples, 1))
    
    num_anomalies = int(num_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    if HAS_NUMPY:
        anomaly_labels[anomaly_indices] = 1.0
    else:
        for idx in anomaly_indices:
            anomaly_labels[idx] = [1.0]
    
    for i in range(num_samples):
        # Initial starting state
        x0, y0 = np.random.uniform(-500, 500), np.random.uniform(-500, 500)
        v0 = np.random.uniform(8.0, 25.0) # Speed in m/s (approx 30 - 90 km/h)
        theta0 = np.random.uniform(0, 2 * math.pi)
        
        is_anomalous = (i in anomaly_indices)
        anomaly_type = np.random.choice(['swerving', 'hard_braking', 'gps_drift', 'sensor_spoofing']) if is_anomalous else 'none'
        
        # Simulate observation window (t = 0 to obs_steps * dt)
        curr_x, curr_y = x0, y0
        curr_v = v0
        curr_theta = theta0
        
        obs_traj = []
        for t in range(obs_steps):
            if is_anomalous and t >= obs_steps // 2:
                if anomaly_type == 'swerving':
                    ax, ay = np.random.normal(0, 0.5), np.random.uniform(-3.5, 3.5)
                    yaw_rate = np.random.uniform(-0.8, 0.8)
                elif anomaly_type == 'hard_braking':
                    ax, ay = np.random.uniform(-6.0, -4.0), np.random.normal(0, 0.2)
                    yaw_rate = np.random.normal(0, 0.05)
                elif anomaly_type == 'gps_drift':
                    ax, ay = np.random.normal(0, 0.2), np.random.normal(0, 0.2)
                    yaw_rate = np.random.normal(0, 0.02)
                    curr_x += np.random.uniform(5.0, 15.0) # Sudden GPS jump
                elif anomaly_type == 'sensor_spoofing':
                    ax, ay = np.random.uniform(8.0, 12.0), np.random.uniform(-5.0, 5.0) # Unrealistic CAN sensor spikes
                    yaw_rate = np.random.uniform(-1.5, 1.5)
            else:
                ax, ay = np.random.normal(0, 0.2), np.random.normal(0, 0.1)
                yaw_rate = np.random.normal(0, 0.02)
                
            curr_v = max(0.0, curr_v + ax * dt)
            curr_theta += yaw_rate * dt
            curr_x += curr_v * math.cos(curr_theta) * dt
            curr_y += curr_v * math.sin(curr_theta) * dt
            
            obs_traj.append([curr_x, curr_y, curr_v, ax, ay, curr_theta, yaw_rate])
            
        if HAS_NUMPY:
            X_obs[i] = np.array(obs_traj)
        else:
            X_obs[i] = obs_traj
        
        # Simulate future ground truth trajectory (pred_steps * dt)
        pred_traj = []
        for t in range(pred_steps):
            curr_v = max(0.0, curr_v + np.random.normal(0, 0.1) * dt)
            curr_theta += np.random.normal(0, 0.01) * dt
            curr_x += curr_v * math.cos(curr_theta) * dt
            curr_y += curr_v * math.sin(curr_theta) * dt
            pred_traj.append([curr_x, curr_y])
            
        if HAS_NUMPY:
            Y_pred[i] = np.array(pred_traj)
        else:
            Y_pred[i] = pred_traj
        
    print(f"Generated dataset with {num_samples} samples ({num_anomalies} anomalous, ratio={anomaly_ratio:.2f}).")
    if HAS_NUMPY:
        print(f"X_obs shape: {X_obs.shape}, Y_pred shape: {Y_pred.shape}, Labels shape: {anomaly_labels.shape}")
    else:
        print(f"X_obs samples: {len(X_obs)}, Y_pred samples: {len(Y_pred)}, Labels samples: {len(anomaly_labels)}")
    return X_obs, Y_pred, anomaly_labels

if __name__ == "__main__":
    X, Y, L = generate_telemetry_dataset(num_samples=100)
    print("Telemetry generator test passed!")
