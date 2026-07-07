import numpy as np
import subprocess
import sys

# 1. Generate a small 10k body dataset
N = 10000
np.random.seed(42)
mass = np.random.uniform(0.5, 1.5, N)
pos = np.random.uniform(-100, 100, (N, 3))
vel = np.zeros((N, 3))

with open("small_test.txt", "w") as f:
    for i in range(N):
        f.write(f"{pos[i][0]},{pos[i][1]},{pos[i][2]},{mass[i]}\n")

# 2. Compute Exact O(N^2) Forces on CPU (Ground Truth)
print("Computing Exact N-Body Forces (Ground Truth)...")
G = 6.674e-11
forces_exact = np.zeros((N, 3))
for i in range(N):
    diff = pos - pos[i]
    dist_sq = np.sum(diff**2, axis=1)
    dist = np.sqrt(dist_sq)
    
    # Avoid division by zero
    mask = dist > 1e-6
    force_mag = G * mass[i] * mass / (dist_sq * dist + 1e-12)
    forces_exact[i] = np.sum(diff[mask] * force_mag[mask, np.newaxis], axis=0)

mag_exact = np.sum(np.linalg.norm(forces_exact, axis=1))
print(f"Exact Total Force Magnitude: {mag_exact:.2e}")

# 3. Run your DSL on the same dataset
print("\nRunning GPRT DSL...")
result = subprocess.run(["./target/release/barneshut", "small_test.txt", "0.5"], capture_output=True, text=True)
print(result.stdout)

# Extract DSL Magnitude from stdout
for line in result.stdout.split('\n'):
    if "Total Force Magnitude" in line:
        dsl_mag = float(line.split(":")[1].strip())
        error = abs(dsl_mag - mag_exact) / mag_exact * 100
        print(f"\n=== VERIFICATION RESULT ===")
        print(f"Relative Error: {error:.2f}%")
        if error < 2.0:
            print("✅ SUCCESS: DSL physics engine is highly accurate!")
        else:
            print("⚠️ WARNING: Significant divergence from exact N-Body.")
