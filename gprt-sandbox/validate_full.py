import numpy as np
from scipy.spatial import cKDTree
import sys
import time

if len(sys.argv) < 4:
    print("Usage: python3 validate_full.py <data.csv> <queries.csv> <k>")
    sys.exit(1)

data_path = sys.argv[1]
query_path = sys.argv[2]
k = int(sys.argv[3])

print(f"Loading datasets...")
data = np.loadtxt(data_path, delimiter=',')
queries = np.loadtxt(query_path, delimiter=',')

print(f"Computing Ground Truth CPU KNN via parallel cKDTree...")
t0 = time.time()
tree = cKDTree(data)
# workers=-1 utilizes all CPU cores for parallel verification
truth_dists, truth_indices = tree.query(queries, k=k, workers=-1)
print(f"CPU Ground Truth generated in {time.time() - t0:.2f} seconds.")

with open('knn_results.txt', 'r') as f:
    lines = f.readlines()

correct = 0
total = 0
tie_breaks_accepted = 0

print("Verifying GPU results...")
for i, line in enumerate(lines):
    if not line.strip(): continue
    total += 1
    
    parts = line.strip().split(':')
    neighbors = eval(parts[1])
    
    gpu_indices = np.array(neighbors)
    truth_ids = truth_indices[i]
    
    if len(gpu_indices) != k:
         continue
         
    # 1. Strict ID Match
    if np.array_equal(gpu_indices, truth_ids):
        correct += 1
    else:
        # 2. Distance Equivalence Match (Resolves arbitrary tie-breaks)
        gpu_dists = np.linalg.norm(data[gpu_indices] - queries[i], axis=1)
        max_truth_dist = truth_dists[i][-1]
        
        if np.all(gpu_dists <= max_truth_dist + 1e-5):
            correct += 1
            tie_breaks_accepted += 1

print(f"\n✅ Validation Accuracy: {correct}/{total} ({correct/total*100:.4f}%)")
if tie_breaks_accepted > 0:
    print(f"   (Note: {tie_breaks_accepted} queries had distance ties and were accepted as correct)")
