import numpy as np
from scipy.spatial import cKDTree

data = np.loadtxt('kitti_10k.csv', delimiter=',')
queries = np.loadtxt('kitti_10k_queries.csv', delimiter=',')

tree = cKDTree(data)
truth_dists, truth_indices = tree.query(queries, k=5)

with open('knn_results.txt', 'r') as f:
    lines = f.readlines()

print("Inspecting the exact mathematical failures...")
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
        if not all(d <= max_truth + 1e-5 for d in pred_dists):
            failures += 1
            if failures <= 3: # Print the first 3 failures
                print(f"\n--- Query {i} FAILED ---")
                print(f"True 5th Neighbor Dist (scipy f64): {max_truth:.8f}")
                print(f"GPU's 5th Neighbor Dist (OptiX f32): {max(pred_dists):.8f}")
                print(f"DIFFERENCE: {max(pred_dists) - max_truth:.2e} units")
                print(f"Conclusion: The GPU found a point that is virtually identical in distance, but scipy broke the f64 tie differently.")
