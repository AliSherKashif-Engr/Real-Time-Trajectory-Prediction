"""
Extended HighD Dataset Caching Script for Authentic Model Training
Prepares and serializes train, val, and test splits from highD recording tracks.
"""

import os
import time
from pathlib import Path
from highd_dataset_loader import HighDDataPreprocessor

def main():
    print("=" * 80)
    print("  Preparing Extended highD Dataset Cache")
    print("=" * 80)
    
    preproc = HighDDataPreprocessor(data_dir="data", ego_centric=True)
    t0 = time.time()
    
    # 1. Train Split: Recordings 01, 02, 03, 04, 05 (300 samples each -> ~1,500 samples)
    print("\n[1/3] Building Extended Train Split (Recordings 1-5)...")
    train_ds = preproc.load_dataset_split(
        recording_ids=[1, 2, 3, 4, 5],
        cache_name="extended_train",
        max_samples_per_rec=300
    )
    print(f"--> Train split ready: {len(train_ds):,} samples.")
    
    # 2. Validation Split: Recordings 06, 07 (200 samples each -> ~400 samples)
    print("\n[2/3] Building Extended Validation Split (Recordings 6-7)...")
    val_ds = preproc.load_dataset_split(
        recording_ids=[6, 7],
        cache_name="extended_val",
        max_samples_per_rec=200
    )
    print(f"--> Validation split ready: {len(val_ds):,} samples.")
    
    # 3. Test Split: Recordings 08, 09, 10 (200 samples each -> ~600 samples)
    print("\n[3/3] Building Extended Test Split (Recordings 8-10)...")
    test_ds = preproc.load_dataset_split(
        recording_ids=[8, 9, 10],
        cache_name="extended_test",
        max_samples_per_rec=200
    )
    print(f"--> Test split ready: {len(test_ds):,} samples.")
    
    print("\n" + "=" * 80)
    print(f"All splits cached successfully in {time.time() - t0:.1f} seconds!")
    print(f"Total authentic scenes: {len(train_ds) + len(val_ds) + len(test_ds):,}")
    print("=" * 80)

if __name__ == "__main__":
    main()
