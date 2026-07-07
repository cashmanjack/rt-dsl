#!/bin/bash
set -e

ARKADE_DIR=~/Arkade
ARKADE_BIN=$ARKADE_DIR/build/sample02-withTrueknn
DATASET_DIR=$ARKADE_DIR/datasets

GPRT_BIN="$HOME/RT_DSL/target/release/gprt-sandbox"
PROFILER_BIN="$HOME/RT_DSL/target/release/gprt-profiler"
GPRT_DATA_DIR="gprt-sandbox/benchmarks/data"
GPRT_QUERY_DIR="gprt-sandbox/benchmarks/queries"

DATASETS=("3droad" "kitti" "3diono" "porto")
K=5

echo "Dataset,N_Points,N_Queries,Arkade_Time_s,GPRT_Time_s,GPRT_Percentile,GPRT_Multiplier" > final_head_to_head.csv

echo "Building GPRT DSL and Profiler in Release Mode..."
(cd ~/RT_DSL && cargo build --release -q)

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Processing $DS (k=$K)"
    echo "========================================="
    
    GPRT_DATA=$GPRT_DATA_DIR/$DS.csv
    GPRT_QUERY=$GPRT_QUERY_DIR/${DS}_queries.csv
    ARKADE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    if [ ! -f "$GPRT_DATA" ]; then
        echo "Warning: $GPRT_DATA not found. Skipping."
        continue
    fi

    # Use Arkade's official counts
    if [ -f "$DATASET_DIR/counts.txt" ]; then
        NPOINTS=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $2}')
        NQUERIES=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $3}')
    else
        NPOINTS=$(wc -l < "$GPRT_DATA")
        NQUERIES=$(wc -l < "$GPRT_QUERY")
    fi

    # STEP 1: Run Profiler
    echo "   [Profiler] Auto-tuning parameters for $DS..."
    rm -f gprt_wisdom.json
    PROFILER_OUT=$($PROFILER_BIN $GPRT_DATA $GPRT_QUERY $K 2>&1)
    
    # Extract chosen parameters from profiler output
    BEST_TIME=$(echo "$PROFILER_OUT" | grep "Best time:" | grep -o -P '(?<=Best time: )[0-9\.]+')
    BEST_P=$(echo "$PROFILER_OUT" | grep -o -P 'P=[0-9\.]+, M=[0-9\.]+' | tail -1 | grep -o -P '(?<=P=)[0-9\.]+')
    BEST_M=$(echo "$PROFILER_OUT" | grep -o -P 'P=[0-9\.]+, M=[0-9\.]+' | tail -1 | grep -o -P '(?<=M=)[0-9\.]+')
    
    echo "   [Profiler] Chose P=$BEST_P, M=$BEST_M (estimated ${BEST_TIME}s)"
    
    # STEP 2: Run GPRT with auto-tuned parameters
    echo "   [GPRT] Running DSL with auto-tuned parameters..."
    GPRT_OUT=$($GPRT_BIN $GPRT_DATA $GPRT_QUERY $K $BEST_P $BEST_M 2>&1)
    GPRT_TIME_MS=$(echo "$GPRT_OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' || true)
    
    if [ -n "$GPRT_TIME_MS" ]; then
        GPRT_TIME_S=$(awk "BEGIN {print $GPRT_TIME_MS/1000}")
    else
        GPRT_TIME_S="N/A"
    fi

    # STEP 3: Run Arkade
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

    echo "   Result -> Arkade: ${ARKADE_TIME}s | GPRT: ${GPRT_TIME_S}s (P=$BEST_P, M=$BEST_M)"
    echo "$DS,$NPOINTS,$NQUERIES,$ARKADE_TIME,$GPRT_TIME_S,$BEST_P,$BEST_M" >> final_head_to_head.csv
done

echo ""
echo "========================================="
echo " FINAL HEAD-TO-HEAD RESULTS (Seconds)"
echo "========================================="
column -s, -t < final_head_to_head.csv
