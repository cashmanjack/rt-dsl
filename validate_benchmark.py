import numpy as np
import sys

data_path = sys.argv[1]
file1_path = sys.argv[2] # gprt-sandbox/knn_results.txt
file2_path = sys.argv[3] # ~/Arkade/build/arkade_results.txt
k = int(sys.argv[4])

print("Loading data...")
data = np.loadtxt(data_path, delimiter=',')

def load_results(path):
    results = []
    with open(path, 'r') as f:
        for line in f:
            if '[' in line and ']' in line:
                ids_str = line.split('[')[1].split(']')[0]
                ids = [int(x) for x in ids_str.split(', ')] if ids_str else []
                results.append(ids)
    return results

print(f"Loading {file1_path} and {file2_path}...")
res1 = load_results(file1_path)
res2 = load_results(file2_path)

matches = 0
total = min(len(res1), len(res2))

for i in range(total):
    ids1 = res1[i]
    ids2 = res2[i]
    if len(ids1) < k or len(ids2) < k:
        continue
        
    pts1 = data[ids1]
    pts2 = data[ids2]
    
    # Compare spatial distances to handle duplicate ID mismatches
    dists1 = np.sqrt(np.sum((pts1 - data[i])**2, axis=1)) 
    dists2 = np.sqrt(np.sum((pts2 - data[i])**2, axis=1))
    
    dists1.sort()
    dists2.sort()
    
    if np.allclose(dists1, dists2, atol=1e-5):
        matches += 1

print(f"\n✅ Engine-vs-Engine Spatial Match: {matches}/{total} ({100*matches/total:.2f}%)")
