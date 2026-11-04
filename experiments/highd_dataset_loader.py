"""
HighD Dataset Loader and Telemetry Feature Preprocessor for MT-STGTD
Author: Ali Sher Kashif (Ph.D. Research, Paper 01)

Extracts authentic multi-modal 7D kinematic vehicular telemetry from raw highD recordings:
  1. Position coordinates (x, y) - normalized in ego-centric frame
  2. Scalar velocity (v = sqrt(vx^2 + vy^2))
  3. Longitudinal acceleration (ax)
  4. Lateral acceleration (ay)
  5. Heading angle (theta = arctan2(vy, vx) relative to ego)
  6. Yaw rate (omega = d(theta)/dt)

Constructs dynamic V2V interaction graphs (pairwise distance matrices) and
implements the 15% controlled anomaly injection protocol matching Section 4.1 of the manuscript.
"""

import os
import glob
import math
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class HighDDataset(Dataset):
    """
    PyTorch Dataset for authentic highD trajectory sequences.
    Ingests 7D kinematics with dynamic spatial graphs and anomaly labels.
    """
    def __init__(self, x_data, y_data, dist_matrices, anomaly_labels, masks, metadata=None):
        """
        Args:
            x_data: FloatTensor [num_samples, max_neighbors, obs_steps, 7]
            y_data: FloatTensor [num_samples, max_neighbors, pred_steps, 2] (in meters, ego frame)
            dist_matrices: FloatTensor [num_samples, obs_steps, max_neighbors, max_neighbors]
            anomaly_labels: FloatTensor [num_samples, max_neighbors, 1]
            masks: FloatTensor [num_samples, max_neighbors] (1: real vehicle, 0: padding)
            metadata: list of dicts (optional)
        """
        self.x_data = torch.as_tensor(x_data, dtype=torch.float32)
        self.y_data = torch.as_tensor(y_data, dtype=torch.float32)
        self.dist_matrices = torch.as_tensor(dist_matrices, dtype=torch.float32)
        self.anomaly_labels = torch.as_tensor(anomaly_labels, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.float32)
        self.metadata = metadata

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return {
            "x": self.x_data[idx],
            "y": self.y_data[idx],
            "dist_matrix": self.dist_matrices[idx],
            "anomaly_label": self.anomaly_labels[idx],
            "mask": self.masks[idx]
        }


class HighDDataPreprocessor:
    """
    Parses raw highD CSV recordings, extracts 7D telemetry, transforms to ego-centric coordinates,
    builds dynamic graphs, and formats train/val/test splits for MT-STGTD.
    """
    def __init__(
        self,
        data_dir="data",
        obs_steps=20,          # 2.0s at 10 Hz
        pred_steps=30,         # 3.0s at 10 Hz
        raw_fps=25,            # highD default frame rate
        target_fps=10,         # 10 Hz benchmark sampling rate
        max_neighbors=6,       # 1 target + up to 5 surrounding neighbors
        neighbor_radius=60.0,  # Max spatial search radius in meters
        stride_frames=50,      # Sliding window stride in raw frames (2.0s)
        anomaly_ratio=0.15,    # 15% anomaly injection for benchmarking
        ego_centric=True,      # Align coordinates with ego-heading for rotational invariance
        seed=42
    ):
        self.data_dir = data_dir
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.raw_fps = raw_fps
        self.target_fps = target_fps
        self.max_neighbors = max_neighbors
        self.neighbor_radius = neighbor_radius
        self.stride_frames = stride_frames
        self.anomaly_ratio = anomaly_ratio
        self.ego_centric = ego_centric
        self.seed = seed
        
        # Calculate raw frame window size
        # Total duration = (obs_steps + pred_steps) / target_fps = 5.0 seconds
        self.total_duration_sec = (obs_steps + pred_steps) / target_fps
        self.total_raw_frames = int(round(self.total_duration_sec * raw_fps)) # 125 frames at 25 Hz
        
        # Resampling indices from raw frames (125) to target steps (50)
        self.resample_indices = np.round(
            np.linspace(0, self.total_raw_frames - 1, obs_steps + pred_steps)
        ).astype(int)
        
        random.seed(seed)
        np.random.seed(seed)

    def _extract_7d_features(self, df_vehicle, dt=0.1):
        """
        Computes 7D kinematic telemetry from vehicle trajectory DataFrame:
        [x, y, v_scalar, ax, ay, heading, yaw_rate]
        """
        x = df_vehicle["x"].values
        y = df_vehicle["y"].values
        vx = df_vehicle["xVelocity"].values
        vy = df_vehicle["yVelocity"].values
        ax = df_vehicle["xAcceleration"].values
        ay = df_vehicle["yAcceleration"].values
        
        v_scalar = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx)
        yaw_rate = np.gradient(np.unwrap(heading), dt)
        
        return np.column_stack([x, y, v_scalar, ax, ay, heading, yaw_rate])

    def _inject_anomaly(self, x_seq, y_seq, anomaly_type):
        """
        Injects controlled physical/sensor anomalies into a single vehicle trajectory sequence.
        x_seq: [T_obs, 7]
        y_seq: [T_pred, 2]
        """
        x_mod = x_seq.copy()
        y_mod = y_seq.copy()
        T_obs = len(x_seq)
        inject_start = T_obs // 2 # Inject starting halfway through observation

        if anomaly_type == "swerving":
            # High lateral acceleration and oscillating yaw rate
            for t in range(inject_start, T_obs):
                swerve_factor = np.sin((t - inject_start) * 1.5)
                x_mod[t, 4] += swerve_factor * 3.5  # ay
                x_mod[t, 6] += swerve_factor * 0.8  # yaw_rate
                x_mod[t, 1] += swerve_factor * 1.8  # y lateral position shift
                
        elif anomaly_type == "hard_braking":
            # Severe longitudinal deceleration
            for t in range(inject_start, T_obs):
                decel = -5.5 + np.random.normal(0, 0.3)
                x_mod[t, 3] = decel                # ax
                x_mod[t, 2] = max(0.0, x_mod[t, 2] - 4.0 * (t - inject_start) * 0.1) # speed drops

        elif anomaly_type == "gps_drift":
            # Sudden coordinate jump and noise
            drift_x = np.random.uniform(8.0, 15.0)
            drift_y = np.random.uniform(4.0, 8.0)
            for t in range(inject_start, T_obs):
                x_mod[t, 0] += drift_x + np.random.normal(0, 0.8)
                x_mod[t, 1] += drift_y + np.random.normal(0, 0.8)

        elif anomaly_type == "sensor_spoofing":
            # Physically inconsistent speed and acceleration spikes
            for t in range(inject_start, T_obs):
                x_mod[t, 2] *= 1.8                 # speed spike
                x_mod[t, 3] = np.random.uniform(-8.0, 8.0) # erratic ax
                x_mod[t, 4] = np.random.uniform(-6.0, 6.0) # erratic ay

        return x_mod, y_mod

    def _to_ego_frame(self, pts_2d, origin_x, origin_y, heading_angle):
        """
        Transforms 2D points to ego-centric coordinates aligned with vehicle heading:
        pts_2d: [N_points, 2] (x, y)
        """
        dx = pts_2d[:, 0] - origin_x
        dy = pts_2d[:, 1] - origin_y
        cos_h = math.cos(heading_angle)
        sin_h = math.sin(heading_angle)
        x_ego = cos_h * dx + sin_h * dy
        y_ego = -sin_h * dx + cos_h * dy
        return np.column_stack([x_ego, y_ego])

    def process_recording(self, recording_id, max_samples=None):
        """
        Loads and extracts trajectory scenes from a specific highD recording CSV.
        Returns:
            samples_x, samples_y, samples_dist, samples_label, samples_mask
        """
        rec_str = f"{recording_id:02d}"
        tracks_file = os.path.join(self.data_dir, f"{rec_str}_tracks.csv")
        
        if not os.path.exists(tracks_file):
            raise FileNotFoundError(f"Missing highD tracks file: {tracks_file}")
            
        usecols = [
            "frame", "id", "x", "y", "xVelocity", "yVelocity", 
            "xAcceleration", "yAcceleration", "laneId"
        ]
        tracks_df = pd.read_csv(tracks_file, usecols=usecols)
        
        # Group by vehicle ID
        veh_groups = {veh_id: group for veh_id, group in tracks_df.groupby("id")}
        
        # Frame lookup dictionary: frame -> list of (veh_id, x, y)
        frame_lookup = tracks_df.groupby("frame")[["id", "x", "y"]].apply(
            lambda g: g.values
        ).to_dict()
        
        samples_x = []
        samples_y = []
        samples_dist = []
        samples_label = []
        samples_mask = []
        
        veh_ids = list(veh_groups.keys())
        
        for v_id in veh_ids:
            v_df = veh_groups[v_id].sort_values("frame").reset_index(drop=True)
            v_len = len(v_df)
            if v_len < self.total_raw_frames:
                continue
                
            # Iterate through sliding windows
            for start_idx in range(0, v_len - self.total_raw_frames + 1, self.stride_frames):
                raw_window = v_df.iloc[start_idx : start_idx + self.total_raw_frames]
                # Downsample to 10 Hz (50 steps)
                downsampled = raw_window.iloc[self.resample_indices].reset_index(drop=True)
                
                # Split observation (20 steps) and prediction (30 steps)
                obs_df = downsampled.iloc[:self.obs_steps]
                pred_df = downsampled.iloc[self.obs_steps:]
                
                # Ego vehicle state at last observation step t=T_obs-1
                origin_x = obs_df.iloc[-1]["x"]
                origin_y = obs_df.iloc[-1]["y"]
                vx_0 = obs_df.iloc[-1]["xVelocity"]
                vy_0 = obs_df.iloc[-1]["yVelocity"]
                ego_heading = math.atan2(vy_0, vx_0) if self.ego_centric else 0.0
                
                # Find neighboring vehicles present at t=T_obs-1
                target_frame = int(obs_df.iloc[-1]["frame"])
                co_vehicles = frame_lookup.get(target_frame, [])
                
                # Rank neighbors by Euclidean distance to target vehicle
                neighbors = []
                for other_row in co_vehicles:
                    other_id = int(other_row[0])
                    if other_id == v_id:
                        continue
                    dx = other_row[1] - origin_x
                    dy = other_row[2] - origin_y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist <= self.neighbor_radius:
                        neighbors.append((dist, other_id))
                        
                neighbors.sort(key=lambda item: item[0])
                selected_neighbor_ids = [n[1] for n in neighbors[:self.max_neighbors - 1]]
                scene_veh_ids = [v_id] + selected_neighbor_ids
                
                # Initialize scene tensors
                scene_x = np.zeros((self.max_neighbors, self.obs_steps, 7), dtype=np.float32)
                scene_y = np.zeros((self.max_neighbors, self.pred_steps, 2), dtype=np.float32)
                scene_mask = np.zeros(self.max_neighbors, dtype=np.float32)
                scene_label = np.zeros((self.max_neighbors, 1), dtype=np.float32)
                
                obs_frames = obs_df["frame"].values
                pred_frames = pred_df["frame"].values
                
                for slot_idx, current_id in enumerate(scene_veh_ids):
                    c_group = veh_groups[current_id]
                    c_obs = c_group[c_group["frame"].isin(obs_frames)].sort_values("frame")
                    
                    if len(c_obs) != self.obs_steps:
                        continue
                        
                    feat_7d = self._extract_7d_features(c_obs, dt=1.0/self.target_fps)
                    
                    # Transform coordinates to ego-centric coordinate frame
                    ego_coords = self._to_ego_frame(feat_7d[:, :2], origin_x, origin_y, ego_heading)
                    feat_7d[:, 0] = ego_coords[:, 0]
                    feat_7d[:, 1] = ego_coords[:, 1]
                    # Relative heading
                    feat_7d[:, 5] = (feat_7d[:, 5] - ego_heading + math.pi) % (2 * math.pi) - math.pi
                    
                    # Future trajectory
                    c_pred = c_group[c_group["frame"].isin(pred_frames)].sort_values("frame")
                    if len(c_pred) == self.pred_steps:
                        pred_raw = np.column_stack([c_pred["x"].values, c_pred["y"].values])
                        pred_coords = self._to_ego_frame(pred_raw, origin_x, origin_y, ego_heading)
                    else:
                        # Constant velocity fallback if recording boundary is reached
                        last_x = feat_7d[-1, 0]
                        last_y = feat_7d[-1, 1]
                        vx_ego = feat_7d[-1, 2] * math.cos(feat_7d[-1, 5])
                        vy_ego = feat_7d[-1, 2] * math.sin(feat_7d[-1, 5])
                        t_steps = np.arange(1, self.pred_steps + 1) * (1.0 / self.target_fps)
                        pred_coords = np.column_stack([last_x + vx_ego * t_steps, last_y + vy_ego * t_steps])
                        
                    # Inject controlled anomaly into target vehicle (slot 0)
                    is_anom = False
                    if slot_idx == 0 and (np.random.rand() < self.anomaly_ratio):
                        anom_type = np.random.choice(["swerving", "hard_braking", "gps_drift", "sensor_spoofing"])
                        feat_7d, pred_coords = self._inject_anomaly(feat_7d, pred_coords, anom_type)
                        is_anom = True
                        
                    scene_x[slot_idx] = feat_7d
                    scene_y[slot_idx] = pred_coords
                    scene_mask[slot_idx] = 1.0
                    scene_label[slot_idx, 0] = 1.0 if is_anom else 0.0
                    
                # Pairwise distance matrix across observation steps
                scene_dist = np.full((self.obs_steps, self.max_neighbors, self.max_neighbors), 1000.0, dtype=np.float32)
                for t in range(self.obs_steps):
                    for i in range(self.max_neighbors):
                        if scene_mask[i] == 0:
                            continue
                        for j in range(self.max_neighbors):
                            if scene_mask[j] == 0:
                                continue
                            if i == j:
                                scene_dist[t, i, j] = 0.0
                            else:
                                dx = scene_x[i, t, 0] - scene_x[j, t, 0]
                                dy = scene_x[i, t, 1] - scene_x[j, t, 1]
                                scene_dist[t, i, j] = math.sqrt(dx**2 + dy**2)
                                
                samples_x.append(scene_x)
                samples_y.append(scene_y)
                samples_dist.append(scene_dist)
                samples_label.append(scene_label)
                samples_mask.append(scene_mask)
                
                if max_samples and len(samples_x) >= max_samples:
                    break
            if max_samples and len(samples_x) >= max_samples:
                break
                
        return (
            np.array(samples_x, dtype=np.float32),
            np.array(samples_y, dtype=np.float32),
            np.array(samples_dist, dtype=np.float32),
            np.array(samples_label, dtype=np.float32),
            np.array(samples_mask, dtype=np.float32)
        )

    def load_dataset_split(self, recording_ids, cache_name=None, max_samples_per_rec=None):
        """
        Loads multiple recordings, concatenates them, and caches the result for rapid reloading.
        """
        cache_file = None
        if cache_name:
            cache_dir = os.path.join(self.data_dir, "processed_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{cache_name}.npz")
            if os.path.exists(cache_file):
                print(f"Loading cached highD split from '{cache_file}'...")
                cached = np.load(cache_file)
                return HighDDataset(
                    cached["x"], cached["y"], cached["dist"], 
                    cached["label"], cached["mask"]
                )
                
        all_x, all_y, all_dist, all_labels, all_masks = [], [], [], [], []
        t0 = time.time()
        for rec_id in recording_ids:
            print(f"Processing highD recording {rec_id:02d}...")
            x_arr, y_arr, dist_arr, lbl_arr, msk_arr = self.process_recording(
                rec_id, max_samples=max_samples_per_rec
            )
            if len(x_arr) > 0:
                all_x.append(x_arr)
                all_y.append(y_arr)
                all_dist.append(dist_arr)
                all_labels.append(lbl_arr)
                all_masks.append(msk_arr)
                
        if not all_x:
            raise ValueError(f"No trajectory samples extracted from recordings: {recording_ids}")
            
        cat_x = np.concatenate(all_x, axis=0)
        cat_y = np.concatenate(all_y, axis=0)
        cat_dist = np.concatenate(all_dist, axis=0)
        cat_labels = np.concatenate(all_labels, axis=0)
        cat_masks = np.concatenate(all_masks, axis=0)
        
        elapsed = time.time() - t0
        print(f"Extraction complete in {elapsed:.1f}s: {len(cat_x):,} total scene samples.")
        
        if cache_file:
            print(f"Saving preprocessed split to '{cache_file}'...")
            np.savez_compressed(
                cache_file,
                x=cat_x, y=cat_y, dist=cat_dist, label=cat_labels, mask=cat_masks
            )
            
        return HighDDataset(cat_x, cat_y, cat_dist, cat_labels, cat_masks)


def get_highd_dataloaders(
    data_dir="data",
    train_recordings=range(1, 41),   # Recordings 01-40 for Training (70%)
    val_recordings=range(41, 51),     # Recordings 41-50 for Validation (15%)
    test_recordings=range(51, 61),    # Recordings 51-60 for Testing (15%)
    batch_size=32,
    max_samples_per_rec=None,
    num_workers=0
):
    """
    Factory function producing PyTorch DataLoaders partitioned by disjoint recording sessions.
    """
    preprocessor = HighDDataPreprocessor(data_dir=data_dir)
    
    train_ds = preprocessor.load_dataset_split(
        train_recordings, cache_name="highd_train_split", max_samples_per_rec=max_samples_per_rec
    )
    val_ds = preprocessor.load_dataset_split(
        val_recordings, cache_name="highd_val_split", max_samples_per_rec=max_samples_per_rec
    )
    test_ds = preprocessor.load_dataset_split(
        test_recordings, cache_name="highd_test_split", max_samples_per_rec=max_samples_per_rec
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("="*80)
    print("Testing Ego-Centric HighD Preprocessor on authentic recording 01...")
    print("="*80)
    preproc = HighDDataPreprocessor(data_dir="data", ego_centric=True)
    dataset = preproc.load_dataset_split(
        recording_ids=[1], cache_name="test_ego_sample_rec01", max_samples_per_rec=100
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    
    print("\nBatch Verification:")
    print("  x shape (7D Telemetry):            ", batch["x"].shape)          # [8, 6, 20, 7]
    print("  y shape (Ground Truth Trajectory): ", batch["y"].shape)          # [8, 6, 30, 2]
    print("  dist_matrix shape (Dynamic Graph): ", batch["dist_matrix"].shape)  # [8, 20, 6, 6]
    print("  anomaly_label shape:               ", batch["anomaly_label"].shape)# [8, 6, 1]
    print("  mask shape:                        ", batch["mask"].shape)         # [8, 6]
    
    # Inspect target vehicle's future coordinates
    target_future_x = batch["y"][0, 0, :, 0].numpy()
    target_future_y = batch["y"][0, 0, :, 1].numpy()
    print("\nEgo-centric Trajectory Progression Sample (Vehicle 0):")
    print(f"  Future Forward x (1s, 2s, 3s): {target_future_x[9]:.2f}m, {target_future_x[19]:.2f}m, {target_future_x[29]:.2f}m")
    print(f"  Future Lateral y (1s, 2s, 3s): {target_future_y[9]:.2f}m, {target_future_y[19]:.2f}m, {target_future_y[29]:.2f}m")
    print("\nEgo-centric preprocessor validated successfully!")
