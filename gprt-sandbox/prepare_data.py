import csv
import ast
import os
import random
import glob

# Try to import numpy for fast binary parsing, fallback to standard library if missing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import struct
    HAS_NUMPY = False

os.makedirs("benchmarks/data", exist_ok=True)
os.makedirs("benchmarks/queries", exist_ok=True)

TARGET_QUERY_COUNT = 400000
MAX_PORTO_POINTS = 2000000 # Cap Porto at 2M to fit in 12GB VRAM alongside 400K query buffers

def save_csv(points, filepath):
    with open(filepath, 'w') as f:
        for p in points:
            f.write(f"{p[0]},{p[1]},{p[2]}\n")

def sample_and_save_queries(filepath, target_count):
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    q_count = min(target_count, len(points))
    queries = random.sample(points, q_count) if len(points) > q_count else points
    q_path = filepath.replace("/data/", "/queries/").replace(".csv", "_queries.csv")
    save_csv(queries, q_path)
    print(f"  -> Saved {q_count} queries to {q_path}")

# ==========================================
# 1. UniformDist
# ==========================================
print("1. Processing UniformDist...")
uniform_points = []
with open('UniformDist.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if len(row) >= 3:
            uniform_points.append([float(row[0]), float(row[1]), float(row[2])])
save_csv(uniform_points, "benchmarks/data/uniform.csv")
sample_and_save_queries("benchmarks/data/uniform.csv", TARGET_QUERY_COUNT)

# ==========================================
# 2. 3DRoad
# ==========================================
print("2. Processing 3DRoad...")
road_points = []
with open('data/3D_spatial_network.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            road_points.append([float(parts[1]), float(parts[2]), float(parts[3])])
save_csv(road_points, "benchmarks/data/3droad.csv")
sample_and_save_queries("benchmarks/data/3droad.csv", TARGET_QUERY_COUNT)

# ==========================================
# 3. Porto (Streaming to avoid RAM OOM)
# ==========================================
print("3. Processing Porto (train.csv)... this may take 5-10 minutes.")
porto_points = []
with open('data/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015/train.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    poly_idx = header.index('POLYLINE')
    
    for row_num, row in enumerate(reader):
        if len(porto_points) >= MAX_PORTO_POINTS:
            break
        if row_num % 100000 == 0:
            print(f"    ...processed {row_num} trips, found {len(porto_points)} points")
        
        poly_str = row[poly_idx]
        if poly_str and poly_str != '[]':
            try:
                coords = ast.literal_eval(poly_str)
                for x, y in coords:
                    porto_points.append([float(x), float(y), 0.0])
                    if len(porto_points) >= MAX_PORTO_POINTS:
                        break
            except:
                pass

save_csv(porto_points, "benchmarks/data/porto.csv")
sample_and_save_queries("benchmarks/data/porto.csv", TARGET_QUERY_COUNT)

# ==========================================
# 4. KITTI (Binary LiDAR Parsing)
# ==========================================
print("4. Processing KITTI...")
kitti_points = []
# Check both possible download locations
bin_files = sorted(glob.glob('data/raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/*.bin'))
if not bin_files:
    bin_files = sorted(glob.glob('data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/*.bin'))

if bin_files:
    print(f"    Found {len(bin_files)} KITTI LiDAR frames. Parsing...")
    for f in bin_files:
        with open(f, 'rb') as fp:
            data = fp.read()
            if HAS_NUMPY:
                pts = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)[:, :3]
                kitti_points.extend(pts.tolist())
            else:
                num_points = len(data) // 16
                for i in range(num_points):
                    x, y, z, _ = struct.unpack_from('ffff', data, i * 16)
                    kitti_points.append([x, y, z])
    
    save_csv(kitti_points, "benchmarks/data/kitti.csv")
    sample_and_save_queries("benchmarks/data/kitti.csv", TARGET_QUERY_COUNT)
else:
    print("    [WARNING] KITTI .bin files not found. Did you run the raw_data_downloader.sh script?")

print("\nData preparation complete! Files saved to benchmarks/")
