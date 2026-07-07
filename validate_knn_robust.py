import numpy as np
from scipy.spatial import cKDTree
import sys

data_path = sys.argv[1]
query_path = sys.argv[2]
k = int(sys.argv[3])

print("Loading data...")
data = np.loadtxt(data_path, delimiter=',')
queries = np.loadtxt(query_path, delimiter=',')

gpu_results = []
with open('knn_results.txt', 'r') as f:
    for line in f:
        ids_str = line.split('[')[1].split(']')[0]
        ids = [int(x) for x in ids_str.split(', ')] if ids_str else []
        gpu_results.append(ids)

print("Computing CPU Ground Truth...")
tree = cKDTree(data)
# Query k+1 to safely handle self-intersections/duplicates
cpu_dists, cpu_ids = tree.query(queries, k=k+1) 

matches = 0
for i in range(len(queries)):
    gpu_ids = gpu_results[i]
    if len(gpu_ids) < k:
        continue
        
    # Calculate actual spatial distances for the GPU's chosen IDs
    gpu_pts = data[gpu_ids]
    gpu_dists = np.sqrt(np.sum((gpu_pts - queries[i])**2, axis=1))
    gpu_dists.sort()
    
    # Get the k smallest distances from the CPU
    cpu_best_dists = cpu_dists[i][:k]
    
    # Compare the mathematical distances, ignoring duplicate ID ambiguities
    if np.allclose(gpu_dists, cpu_best_dists, atol=1e-5):
        matches += 1

print(f"\n✅ Robust Spatial Validation: {matches}/{len(queries)} ({100*matches/len(queries):.2f}%)")
print("This script validates the actual mathematical distances, ignoring duplicate ID ambiguities.")
