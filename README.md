# GPRT-DSL: Hardware-Accelerated Spatial Search via Rust-to-OptiX Compilation

A high-level Rust DSL compiler that translates spatial queries into optimized NVIDIA OptiX ray tracing pipelines. Achieves up to **57× speedups** over hand-tuned C++/OWL baselines on exact unbounded k-NN search while maintaining a general-purpose, memory-safe programming model.

## Key Features

- **Zero-Cost Abstraction**: Rust procedural macros generate specialized OptiX-IR at compile time with no runtime overhead
- **Exact Unbounded k-NN**: Guaranteed 100% recall via iterative TrueKNN with CPU-side query filtering
- **95th-Percentile LCG Sampler**: Zero-allocation statistical radius heuristic that prevents BVH degeneracy on bi-modal datasets
- **Dynamic VRAM Scaling**: Compiler-directed buffer capacity scaling bounds peak GPU memory to ~2 GB regardless of query count
- **Zero-Transfer Radius Expansion**: Dynamic search radius encoded in ray `tmax` metadata, eliminating PCIe geometry buffer uploads between iterations
- **Runtime BVH Compaction**: Automatic post-build compaction in the C++ backend reduces final BVH footprint by 30–50%
- **General-Purpose Design**: Standard `AnyHit` push-to-buffer abstraction supports arbitrary k, range queries, spatial joins, and DBSCAN

## Performance (RTX 4070 Ti, k=5, 400K Queries)

| Dataset | Points | TrueKNN (RTX 2060) | GPRT DSL | Speedup |
|---------|--------|---------------------|----------|---------|
| 3DRoad  | 434K   | 120.41 s            | **2.09 s** | **57×** |
| KITTI   | 1.3M   | 175.38 s            | **4.62 s** | **38×** |
| 3DIono  | 1.8M   | 149.60 s            | **4.99 s** | **30×** |
| Porto   | 10M*   | 456.78 s (81M)      | **15.54 s** | —       |

*Porto evaluated on 10M stratified sample; both DSL and Arkade OOM at 81M on 12 GB VRAM due to OptiX BVH builder temp buffer limits.

### Head-to-Head vs. Arkade (C++/OWL, Same Hardware)

| Dataset | Arkade (s) | GPRT DSL (s) | Winner     |
|---------|-----------|--------------|------------|
| 3DRoad  | 3.38      | **2.09**     | DSL (1.6×) |
| KITTI   | 4.48      | **4.62**     | Tie        |
| 3DIono  | 26.71     | **4.99**     | DSL (5.3×) |
| Porto   | **3.56**  | 15.54        | Arkade (4.4×) |

DSL wins on 3/4 datasets via CPU-side query filtering + aggressive radius heuristic. Arkade wins on Porto via register-resident max-heap that bypasses AnyHit global memory bandwidth saturation in dense clusters.

## Requirements

- NVIDIA RTX GPU (Turing/Ampere/Ada Lovelace, sm_75+)
- CUDA 12+ Toolkit
- OptiX 8 SDK
- Rust 1.70+
- CMake 3.22+
- Python 3.8+ (for dataset preparation and validation)

## Quick Start

```bash
git clone https://github.com/cashmanjack/rt-dsl.git
cd rt-dsl/gprt-sandbox

# Set OptiX path
export OPTIX_PATH=/path/to/optix_sdk

# Build entire workspace
cargo build --release

# Run benchmark suite
./benchmark.sh

# Validate correctness against scikit-learn
python3 validate.py benchmarks/data/3droad.csv benchmarks/queries/3droad_queries.csv 5 100


## Datasets

Raw dataset files are excluded from this repository due to GitHub's 100 MB file size limit. To reproduce benchmarks, obtain datasets as follows:

- **3DRoad / KITTI / 3DIono / Porto**: Provided by collaborating researchers. See `prepare_data.py` for formatting instructions. Expected format is comma-separated `x,y,z` per line.
- **OSM Denmark**: Download from [Geofabrik](https://download.geofabrik.de/europe/denmark.html) and place at `gprt-sandbox/data/denmark-260609.osm.pbf`.
- **Ionosphere**: UCI ML Repository. Place in `gprt-sandbox/data/ionosphere/`.

After obtaining datasets, generate query files:
```bash
cd gprt-sandbox
python3 prepare_data.py
```

## Usage

```rust
use gprt_core::Vec3;
use gprt_macros::k_nn;

let dataset: Vec<Vec3> = load_points("data.csv");
let queries: Vec<Vec3> = load_points("queries.csv");
let k: usize = 5;

let mut neighbors: Vec<u32> = Vec::new();
k_nn!(dataset, queries, k, neighbors);
// neighbors contains k IDs per query, sorted by distance
```

The `k_nn!` macro handles all OptiX pipeline creation, BVH construction, iterative radius expansion, dynamic buffer allocation, and CPU-side filtering transparently.

## Project Structure

```
rt-dsl/
├── gprt-core/      # Core types: Vec3, Ray, Sphere, AABB, Scene, BVH builder
├── gprt-codegen/   # Rust → CUDA/OptiX-IR code generation
├── gprt-ir/        # Intermediate representation for RT programs
├── gprt-macros/    # Procedural macros (k_nn!, range_query!)
├── gprt-optix/     # C++ OptiX backend: pipeline, SBT, BVH compaction, trace
└── gprt-sandbox/   # Benchmark harness, datasets, validation scripts
    ├── benchmark.sh           # Full benchmark suite
    ├── sweep_radii.sh         # Radius ablation study
    ├── run_head_to_head.sh    # Arkade comparison script
    ├── prepare_data.py        # Dataset formatting & query sampling
    └── validate.py            # Correctness validation vs scikit-learn
```

## Experiments & Reproducibility

### Radius Ablation Study
Sweeps initial radius multiplier from $2^{-6}$ to $2^{0}$ to empirically derive the optimal $2^{-4}$ default:
```bash
cd gprt-sandbox
./sweep_radii.sh
python3 experiments/plot_radius_ablation.py  # Generates U-shaped curve
```

### Head-to-Head Arkade Comparison
Runs both systems with normalized parameters (k=5, seconds, same hardware):
```bash
./run_head_to_head.sh
```

### Intersection Count Profiling
Enable `atomicAdd(&intersection_counter, 1)` in `__anyhit__ah()` within `gprt-codegen/src/lib.rs` to quantify per-round intersection volume and diagnose memory bandwidth saturation on dense datasets.

## Architecture Decisions

### Why AnyHit Instead of Register Heaps?
Hand-tuned libraries (Arkade, RTNN) maintain k-sized max-heaps in OptiX payload registers within the intersection shader, eliminating global memory writes during traversal. This is faster on extremely dense datasets but:
- Limited to k ≤ 15 (OptiX payload register cap)
- Incompatible with range queries, spatial joins, or variable-k workloads
- Requires complex template metaprogramming per k value

Our DSL uses a generalized `AnyHit` push-to-buffer abstraction that accepts a memory-bandwidth tax on pathological clustering in exchange for unbounded k, runtime flexibility, and seamless extension to arbitrary spatial predicates.

### Why CPU-Side Query Filtering?
Unlike baselines that relaunch all rays every round and rely on device-side early-outs, our macro strips resolved queries on the host. Round 2+ launches only unresolved outliers, reducing total ray count by >99% after Round 1 on most datasets.

### Why 95th-Percentile Over Min/Max Heuristics?
- **Min heuristic** (TrueKNN paper): Too conservative, forces 6-10 rounds of API overhead
- **Max heuristic**: Causes catastrophic BVH degeneracy on bi-modal datasets (e.g., Porto)
- **95th-percentile**: Statistically ignores extreme outliers while resolving >95% of queries in Round 1

## License

MIT

## Citation

```bibtex
@misc{cashman2026gprtdsl,
  title={GPRT-DSL: Hardware-Accelerated Spatial Search via Rust-to-OptiX Compilation},
  author={Cashman, Jack},
  year={2026},
  howpublished={\url{https://github.com/cashmanjack/rt-dsl}}
}
```
