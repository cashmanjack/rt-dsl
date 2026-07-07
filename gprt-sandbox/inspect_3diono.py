import numpy as np
from scipy.spatial import cKDTree

data = np.loadtxt('subsets/3diono_10k.csv', delimiter=',')
queries = np.loadtxt('subsets/3diono_10k_queries.csv', delimiter=',')

tree = cKDTree(data)
truth_dists, truth_indices = tree.query(queries, k=5)

with open('knn_results.txt', 'r') as f:
    lines = f.readlines()

print("Inspecting 3diono failures...")
failures = 0
for i, line in enumerate(lines):
    if not line.strip(): continue
    pred_str = line.split(':')[1].strip().strip('[]')
    if not pred_str: continue
    pred = [int(x.strip()) for x in pred_str.split(',')]
    
    t_ids = truth_indices[i].tolist()
    if pred != t_ids:
        pred_dists = []
        for pid in pred:
            diff = data[pid] - queries[i]
            pred_dists.append(np.linalg.norm(diff))
            
        max_truth = truth_dists[i][-1]
        failures += 1
        if failures <= 3:
            print(f"\n--- Query {i} FAILED ---")
            print(f"True 5th Neighbor Dist: {max_truth:.6f}")
            print(f"GPU's 5th Neighbor Dist: {max(pred_dists):.6f}")
            print(f"DIFFERENCE: {max(pred_dists) - max_truth:.2e} units")
            print(f"True IDs: {t_ids}")
            print(f"Pred IDs: {pred}")
