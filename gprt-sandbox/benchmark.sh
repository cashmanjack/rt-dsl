#!/bin/bash
set -e

echo "Building GPRT DSL in Release Mode..."
cargo build --release

BIN="./target/release/gprt-sandbox"
OUT_FILE="benchmark_results.csv"

# CSV Header
echo "Dataset,NumQueries,K,Time_ms,Total_Neighbors" > $OUT_FILE

# Datasets to benchmark

declare -a DATASETS=("uniform" "3droad" "kitti" "porto" "3diono")
K=5

echo "Starting TrueKNN Benchmarks (K=$K)..."
for DS in "${DATASETS[@]}"; do
    DATA_FILE="benchmarks/data/${DS}.csv"
    QUERY_FILE="benchmarks/queries/${DS}_queries.csv"
    
    if [ ! -f "$DATA_FILE" ]; then
        echo "Warning: $DATA_FILE not found. Skipping."
        continue
    fi
    
    # Run binary and append output directly to CSV
    $BIN $DATA_FILE $QUERY_FILE $K >> $OUT_FILE
done

echo -e "\nBenchmark Complete! Results:"
column -s, -t < $OUT_FILE
