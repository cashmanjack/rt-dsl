import random
import sys

def generate_dataset(num_points, filename):
    with open(filename, 'w') as f:
        for _ in range(num_points):
            x = random.uniform(-1000, 1000)
            y = random.uniform(-1000, 1000)
            z = random.uniform(-1000, 1000)
            f.write(f"{x:.4f}, {y:.4f}, {z:.4f}\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    generate_dataset(n, "uniform_100k.csv")
    print(f"Generated {n} points in uniform_100k.csv")
