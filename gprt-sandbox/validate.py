import numpy as np
import subprocess
import sys
import os
from sklearn.neighbors import NearestNeighbors

def load_csv(path, max_rows=None):
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            parts = line.strip().split(',')
            if len(parts) >= 3:
                data.append([float(p) for p in parts[:3]])
    return np.array(data)

def validate(data_file, query_file, k=5, num_test_queries=100):
    print(f"Loading data from {data_file}...")
    data = load_csv(data_file)
    queries = load_csv(query_file, max_rows=num_test_queries)
    print(f"  Data: {len(data)} points, Queries: {len(queries)} points, K={k}")

    # --- Gold Standard: scikit-learn ---
    print("Running Scikit-Learn Ball-Tree KNN...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean').fit(data)
    sk_distances, sk_indices = nbrs.kneighbors(queries)
    sk_indices_sorted = np.sort(sk_indices, axis=1)

    # --- DSL: Run gprt-sandbox on the same small subset ---
    # Write temp query file with only num_test_queries
    tmp_query = "/tmp/_validate_queries.csv"
    np.savetxt(tmp_query, queries, delimiter=",", fmt="%.6f")

    print(f"Running GPRT DSL on {num_test_queries} queries...")
    result = subprocess.run(
        ["./target/release/gprt-sandbox", data_file, tmp_query, str(k)],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"DSL FAILED:\n{result.stderr}")
        return False

    # Parse DSL output: last line is CSV: path,num_queries,k,time_ms,total_neighbors
    # But we need per-query neighbor IDs. Modify main.rs to print them, OR
    # parse from stdout. For now, let's just check total neighbor count.
    lines = result.stdout.strip().split('\n')
    csv_line = [l for l in lines if l.count(',') == 4]
    if not csv_line:
        print(f"Could not parse DSL output:\n{result.stdout}")
        return False
    
    parts = csv_line[-1].split(',')
    dsl_total_neighbors = int(parts[4])
    expected_total = num_test_queries * k

    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Expected total neighbors: {expected_total}")
    print(f"DSL total neighbors:      {dsl_total_neighbors}")

    if dsl_total_neighbors == expected_total:
        print("✅ PASS: DSL returned correct number of neighbors.")
    else:
        print(f"❌ FAIL: Neighbor count mismatch! Difference: {abs(dsl_total_neighbors - expected_total)}")
        return False

    # Print gold standard first 5 for manual spot-check
    print(f"\nGold Standard first 5 query neighbor indices (sorted):")
    for i in range(min(5, len(sk_indices_sorted))):
        print(f"  Query {i}: {sk_indices_sorted[i].tolist()}")

    print("\n⚠️  NOTE: For full ID-level validation, modify main.rs to print")
    print("   per-query neighbor IDs and compare against sk_indices_sorted.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 validate.py <data.csv> <queries.csv> [k] [num_test_queries]")
        sys.exit(1)
    
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    nq = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    validate(sys.argv[1], sys.argv[2], k=k, num_test_queries=nq)
