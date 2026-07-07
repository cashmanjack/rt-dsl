#!/bin/bash
set -e

ARKADE_DIR=~/Arkade
ARKADE_BIN=$ARKADE_DIR/build/sample02-withTrueknn
DATASET_DIR=$ARKADE_DIR/datasets

# FIX: Point to the WORKSPACE target directory (where cargo actually builds now)
GPRT_BIN="$HOME/RT_DSL/target/release/gprt-sandbox"
GPRT_DATA_DIR="benchmarks/data"
GPRT_QUERY_DIR="benchmarks/queries"

# Constants
MULTIPLIER=3.0
DATASETS=("3droad" "kitti" "3diono" "porto")
K=5

echo "Dataset,N_Points,N_Queries,Arkade_Time_s,GPRT_Time_s" > final_head_to_head.csv

echo "Building GPRT DSL in Release Mode..."
# Build from the workspace root to ensure the correct binary is updated
(cd ~/RT_DSL && cargo build --release -q -p gprt-sandbox)

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Benchmarking $DS (k=$K)"
    echo "========================================="
    
    # DYNAMIC PERCENTILE SELECTION (Prevents Porto hang & KITTI Zombie Horde)
    if [ "$DS" == "porto" ]; then
        PERCENTILE=0.50  # Porto is a 2D manifold; 0.10 causes 10M-point overlap
    else
        PERCENTILE=0.10  # KITTI, 3droad, 3diono prefer a tight start
    fi
    
    GPRT_DATA=$GPRT_DATA_DIR/$DS.csv
    GPRT_QUERY=$GPRT_QUERY_DIR/${DS}_queries.csv
    ARKADE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    if [ ! -f "$GPRT_DATA" ]; then
        echo "Warning: $GPRT_DATA not found. Skipping."
        continue
    fi

    # Use Arkade's official counts to prevent "Insufficient file size" crash
    if [ -f "$DATASET_DIR/counts.txt" ]; then
        NPOINTS=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $2}')
        NQUERIES=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $3}')
    else
        NPOINTS=$(wc -l < "$GPRT_DATA")
        NQUERIES=$(wc -l < "$GPRT_QUERY")
    fi

    # 1. Run Arkade with Robust Inline Python Estimation
    ARKADE_TIME="N/A"
    if [ -f "$ARKADE_BIN" ] && [ -f "$ARKADE_COMB" ]; then
        echo "   [Arkade] Sampling 10k points for radius estimation..."
        shuf -n 10000 "$ARKADE_DATA" > /tmp/arkade_sample.txt
        
        RADIUS=$(python3 -c "
import numpy as np
from scipy.spatial import cKDTree
X = np.loadtxt('/tmp/arkade_sample.txt')
tree = cKDTree(X)
dists, _ = tree.query(X[:1000], k=5)
print(np.percentile(dists[:, -1], 10))
" 2>&1)

        if [[ "$RADIUS" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
            echo "   [Arkade] Estimated Radius: $RADIUS"
            echo "   [Arkade] Running C++ Binary..."
            ARKADE_OUT=$($ARKADE_BIN $ARKADE_COMB $NPOINTS $NQUERIES $RADIUS 2>&1 || true)
            ARKADE_TIME=$(echo "$ARKADE_OUT" | grep -o -E "[0-9]+\.[0-9]+" | tail -n 1)
            [ -z "$ARKADE_TIME" ] && ARKADE_TIME="N/A"
        else
            ARKADE_TIME="N/A"
            echo "   [Arkade] Warning: Python radius estimation failed."
        fi
    else
        echo "   [Arkade] Missing binary or dataset files. Skipping."
    fi

    # 2. Run GPRT with Tuned Parameters
    echo "   [GPRT] Running DSL (P=$PERCENTILE, M=$MULTIPLIER)..."
    GPRT_OUT=$($GPRT_BIN $GPRT_DATA $GPRT_QUERY $K $PERCENTILE $MULTIPLIER 2>&1)
    GPRT_TIME_MS=$(echo "$GPRT_OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' || true)
    
    if [ -n "$GPRT_TIME_MS" ]; then
        GPRT_TIME_S=$(awk "BEGIN {print $GPRT_TIME_MS/1000}")
    else
        GPRT_TIME_S="N/A"
        echo "   [GPRT] Warning: Failed to extract search_ms."
    fi

    echo "   Result -> Arkade: ${ARKADE_TIME}s | GPRT: ${GPRT_TIME_S}s"
    echo "$DS,$NPOINTS,$NQUERIES,$ARKADE_TIME,$GPRT_TIME_S" >> final_head_to_head.csv
done

echo ""
echo "========================================="
echo " FINAL HEAD-TO-HEAD RESULTS (Seconds)"
echo "========================================="
column -s, -t < final_head_to_head.csv
