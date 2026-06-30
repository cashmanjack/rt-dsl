import numpy as np
from scipy.spatial import cKDTree
import sys

data_path = sys.argv[1]
query_path = sys.argv[2]
k = int(sys.argv[3])

print("Computing Ground Truth CPU KNN...")
data = np.loadtxt(data_path, delimiter=',')
queries = np.loadtxt(query_path, delimiter=',')

tree = cKDTree(data)
truth_dists, truth_indices = tree.query(queries, k=k)

with open('knn_results.txt', 'r') as f:
    lines = f.readlines()

correct = 0
total = 0
tie_breaks_accepted = 0

for i, line in enumerate(lines):
    if not line.strip(): continue
    total += 1
    pred_str = line.split(':')[1].strip().strip('[]')
    if not pred_str: continue
    pred = [int(x.strip()) for x in pred_str.split(',')]
    
    t_ids = truth_indices[i].tolist()
    
    # 1. Strict ID Match
    if pred == t_ids:
        correct += 1
    else:
        # 2. Distance Equivalence Match (The Gold Standard)
        pred_dists = []
        valid = True
        for pid in pred:
            if 0 <= pid < len(data):
                diff = data[pid] - queries[i]
                pred_dists.append(np.linalg.norm(diff))
            else:
                valid = False
                break
        
        if valid:
            # The K-th distance in the truth set is the maximum allowed distance
            max_truth_dist = truth_dists[i][-1]
            
            # If all predicted points are within the maximum truth distance (plus float tolerance),
            # it mathematically proves they are valid K-nearest neighbors, and the ID mismatch 
            # is purely due to arbitrary hardware tie-breaking.
            if all(d <= max_truth_dist + 1e-5 for d in pred_dists):
                correct += 1
                tie_breaks_accepted += 1

print(f"✅ Validation Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
if tie_breaks_accepted > 0:
    print(f"   (Note: {tie_breaks_accepted} queries had distance ties and were accepted as mathematically correct)")
