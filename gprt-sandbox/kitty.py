import numpy as np
import glob
import os
import random

os.makedirs("data", exist_ok=True)
os.makedirs("data/queries", exist_ok=True)

# Find the .bin files from the drive you already downloaded (0009)
bin_files = sorted(glob.glob('data/2011_09_26/*/velodyne_points/data/*.bin'))

if not bin_files:
    print("Error: No .bin files found. Did the extraction finish?")
    exit(1)

print(f"Found {len(bin_files)} LiDAR frames in Drive 0009.")
print("Parsing first 10 frames (~1.15 Million points)...")

kitti_points = []
# 10 frames * ~115,000 points = ~1.15M points
for f in bin_files[:10]: 
    # KITTI binary format: x, y, z, reflectivity (float32)
    points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3]
    kitti_points.extend(points.tolist())

print(f"Parsed {len(kitti_points)} points. Saving to CSV...")
with open("benchmarks/data/kitti.csv", 'w') as f:
    for p in kitti_points:
        f.write(f"{p[0]},{p[1]},{p[2]}\n")

# Sample 400K queries
q_count = min(400000, len(kitti_points))
queries = random.sample(kitti_points, q_count)
with open("benchmarks/queries/kitti_queries.csv", 'w') as f:
    for p in queries:
        f.write(f"{p[0]},{p[1]},{p[2]}\n")

print(f"Success! Saved {len(kitti_points)} data points and {q_count} queries for KITTI.")
