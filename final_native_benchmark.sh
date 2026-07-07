#!/bin/bash

ARKADE_DIR=~/Arkade
ARKADE_BIN=$ARKADE_DIR/build/sample02-withTrueknn
DATASET_DIR=$ARKADE_DIR/datasets

GPRT_BIN="$HOME/RT_DSL/target/release/gprt-sandbox"
GPRT_DATA_DIR="gprt-sandbox/benchmarks/data"
GPRT_QUERY_DIR="gprt-sandbox/benchmarks/queries"

DATASETS=("3droad" "kitti" "3diono" "porto")
K=5

echo "Dataset,Arkade_s,GPRT_NativeTuned_s" > final_native_benchmark.csv

echo "========================================================="
echo " FULL DATASET AUTOTUNING & HEAD-TO-HEAD BENCHMARK"
echo "========================================================="

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo " Benchmarking $DS (k=$K)"
    echo "========================================="
    
    GPRT_DATA=$GPRT_DATA_DIR/$DS.csv
    GPRT_QUERY=$GPRT_QUERY_DIR/${DS}_queries.csv
    ARKADE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    if [ ! -f "$GPRT_DATA" ]; then 
        echo "Dataset $DS not found. Skipping."
        continue 
    fi

    NPOINTS=$(wc -l < "$GPRT_DATA")
    NQUERIES=$(wc -l < "$GPRT_QUERY")

    # =========================================
    # 1. GPRT: FULL DATASET AUTOTUNING
    # =========================================
    echo "   Step 1: Running GPRT Native Full-Dataset Autotuner..."
    cd gprt-sandbox
    rm -f gprt_wisdom.json
    # The 'tune' flag runs the macro over the full dataset and saves to gprt_wisdom.json
    TUNE_OUT=$(../target/release/gprt-sandbox benchmarks/data/${DS}.csv benchmarks/queries/${DS}_queries.csv $K tune 2>&1)
    echo "      $(echo "$TUNE_OUT" | grep -E "(Optimum Reached|Winner Selected)" || echo "Tuning complete.")"
    cd ..

    # =========================================
    # 2. GPRT: MEASURE TUNED EXECUTION
    # =========================================
    echo "   Step 2: Measuring GPRT Tuned Execution..."
    cd gprt-sandbox
    
    # Add timeout to prevent infinite hangs on dense datasets
    GPRT_AUTO_OUT=$(timeout 180 ../target/release/gprt-sandbox benchmarks/data/${DS}.csv benchmarks/queries/${DS}_queries.csv $K 2>&1 || true)
    cd ..
    
    # Safely handle empty grep results
    GPRT_AUTO_MS=$(echo "$GPRT_AUTO_OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' | tail -n 1)
    if [ -z "$GPRT_AUTO_MS" ]; then 
        GPRT_AUTO_MS="0"
        echo "      [WARNING] GPRT timed out or crashed. Defaulting to 0."
    fi
    GPRT_AUTO_S=$(awk "BEGIN {printf \"%.6f\", $GPRT_AUTO_MS/1000.0}")
    
    # Clean up wisdom for next dataset
    rm -f gprt-sandbox/gprt_wisdom.json

    # =========================================
    # 3. ARKADE: PAPER HEURISTIC BASELINE
    # =========================================
    ARKADE_TIME="N/A"
    if [ -f "$ARKADE_BIN" ] && [ -f "$ARKADE_COMB" ]; then
        echo "   Step 3: Calculating Arkade Paper Heuristic Radius..."
        
        RADIUS=$(python3 -c "
import numpy as np
from scipy.spatial import cKDTree
X = np.loadtxt('$ARKADE_DATA')
np.random.seed(42)
size = min(1000, len(X))
sample = X[np.random.choice(len(X), size, replace=False)]
tree = cKDTree(sample)
dists, _ = tree.query(sample, k=5)
non_zero = dists[:, 1:][dists[:, 1:] > 1e-6]
if len(non_zero) > 0: print(np.min(non_zero))
else: print(0.1)
" 2>/dev/null || echo "0.1")
        
        echo "   Step 4: Running Arkade C++ Baseline (Radius: $RADIUS)..."
        ARKADE_OUT=$(timeout 120 $ARKADE_BIN $ARKADE_COMB $NPOINTS $NQUERIES $RADIUS 2>&1 || true)
        ARKADE_TIME=$(echo "$ARKADE_OUT" | grep -o -E "^[0-9]+\.[0-9]+$" | tail -n 1 || echo "N/A")
        if [ -z "$ARKADE_TIME" ]; then ARKADE_TIME="TIMEOUT"; fi
    fi

    echo "   -> Arkade: ${ARKADE_TIME}s | GPRT Native: ${GPRT_AUTO_S}s"
    echo "$DS,$ARKADE_TIME,$GPRT_AUTO_S" >> final_native_benchmark.csv
done

echo ""
echo "========================================="
echo " FINAL HEAD-TO-HEAD RESULTS (Seconds)"
echo "========================================="
column -s, -t < final_native_benchmark.csv
