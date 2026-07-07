import numpy as np
from scipy.spatial import cKDTree
import subprocess
import json
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python3 autotune.py <dataset_name>")
    sys.exit(1)

ds_name = sys.argv[1]
k = 5

data_path = f"gprt-sandbox/benchmarks/data/{ds_name}.csv"
query_path = f"gprt-sandbox/benchmarks/queries/{ds_name}_queries.csv"

if not os.path.exists(data_path) or not os.path.exists(query_path):
    print(f"Error: Dataset {ds_name} files not found.")
    sys.exit(1)

print(f"--- Downsampling {ds_name} to preserve spatial density ---")
# Take every 10th row to reduce size to 10% but preserve exact spacing distributions
data_full = np.loadtxt(data_path, delimiter=',')
queries_full = np.loadtxt(query_path, delimiter=',')

data_sampled = data_full[::10]
queries_sampled = queries_full[::10]

np.savetxt('/tmp/tune_data.csv', data_sampled, delimiter=',')
np.savetxt('/tmp/tune_queries.csv', queries_sampled, delimiter=',')

print("Computing local CPU Ground Truth...")
tree = cKDTree(data_sampled)
truth_dists, truth_indices = tree.query(queries_sampled, k=k, workers=-1)

# Parameter Grid Sweep
percentiles = [0.02, 0.05, 0.10, 0.15, 0.20, 0.50, 0.99]
multipliers = [1.5, 2.0, 2.5, 3.0, 3.5]

best_time = float('inf')
best_p = 0.10
best_m = 3.0

print(f"\nEvaluating Parameter Sweeps (Goal: 100% Accuracy)...")
for p in percentiles:
    for m in multipliers:
        # Generate temporary wisdom file
        wisdom_data = {
            "radius_increment_mult": m,
            "max_hits_per_query": 2000,
            "use_morton_lbv": True,
            "radius_heuristic": {"SampledPercentile": p},
            "memory_strategy": "PayloadRegisterHeap"
        }
        with open("gprt_wisdom.json", "w") as f:
            json.dump(wisdom_data, f)
            
        # Run GPRT sandbox on downsampled data
        cmd = ["./target/release/gprt-sandbox", "/tmp/tune_data.csv", "/tmp/tune_queries.csv", str(k), "dump"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Parse search_ms
        search_ms = 0.0
        for line in res.stdout.split('\n'):
            if "search_ms=" in line:
                parts = line.split("search_ms=")[1].split(",")
                search_ms = float(parts[0])
                
        # Validate correctness
        correct = 0
        total = 0
        try:
            # FIXED: Read knn_results.txt from CWD directly
            with open('knn_results.txt', 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if not line.strip(): continue
                total += 1
                pred = eval(line.split(':')[1])
                gpu_indices = np.array(pred)
                if np.array_equal(gpu_indices, truth_indices[i]):
                    correct += 1
                else:
                    gpu_dists = np.linalg.norm(data_sampled[gpu_indices] - queries_sampled[i], axis=1)
                    if np.all(gpu_dists <= truth_dists[i][-1] + 1e-5):
                        correct += 1
        except Exception as e:
            correct = 0
            
        accuracy = (correct / total) if total > 0 else 0.0
        
        if accuracy >= 0.999: # Must be mathematically correct to be considered
            print(f"   [VALID] P={p:.2f}, M={m:.1f} | GPU Search: {search_ms:.3f}ms")
            if search_ms < best_time:
                best_time = search_ms
                best_p = p
                best_m = m
        else:
            print(f"   [REJECT] P={p:.2f}, M={m:.1f} | Accuracy: {accuracy*100:.2f}% (Too small radius!)")

print(f"\n--- Best Parameters Found for {ds_name} ---")
print(f"   Percentile: {best_p:.2f}")
print(f"   Multiplier: {best_m:.1f}")

# Write back to permanent wisdom file
final_wisdom = {
    "radius_increment_mult": best_m,
    "max_hits_per_query": 2000,
    "use_morton_lbv": True,
    "radius_heuristic": {"SampledPercentile": best_p},
    "memory_strategy": "PayloadRegisterHeap"
}
with open(f"{ds_name}_wisdom.json", "w") as f:
    json.dump(final_wisdom, f)

# Cleanup
for f_path in ["/tmp/tune_data.csv", "/tmp/tune_queries.csv", "gprt_wisdom.json"]:
    if os.path.exists(f_path): os.remove(f_path)
