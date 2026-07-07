#!/bin/bash
set -e

ARKADE_DIR=~/Arkade
ARKADE_BIN=$ARKADE_DIR/build/sample02-withTrueknn
DATASET_DIR=$ARKADE_DIR/datasets

GPRT_BIN="$HOME/RT_DSL/target/release/gprt-sandbox"
GPRT_DATA_DIR="gprt-sandbox/benchmarks/data"
GPRT_QUERY_DIR="gprt-sandbox/benchmarks/queries"

# Fixed good parameters
PERCENTILE=0.10
MULTIPLIER=3.0
K=5

DATASETS=("3droad" "kitti" "3diono" "porto")

echo "Dataset,N_Points,N_Queries,Arkade_Time_s,GPRT_Time_s" > simple_results.csv

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Testing $DS (k=$K, P=$PERCENTILE, M=$MULTIPLIER)"
    echo "========================================="
    
    GPRT_DATA=$GPRT_DATA_DIR/$DS.csv
    GPRT_QUERY=$GPRT_QUERY_DIR/${DS}_queries.csv
    ARKADE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    if [ ! -f "$GPRT_DATA" ]; then
        echo "Warning: $GPRT_DATA not found. Skipping."
        continue
    fi

    # Get counts
    if [ -f "$DATASET_DIR/counts.txt" ]; then
        NPOINTS=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $2}')
        NQUERIES=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $3}')
    else
        NPOINTS=$(wc -l < "$GPRT_DATA")
        NQUERIES=$(wc -l < "$GPRT_QUERY")
    fi

    # Run GPRT
    echo "   Running GPRT..."
    GPRT_OUT=$($GPRT_BIN $GPRT_DATA $GPRT_QUERY $K $PERCENTILE $MULTIPLIER 2>&1 || true)
    GPRT_TIME_MS=$(echo "$GPRT_OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' || echo "0")
    GPRT_TIME_S=$(awk "BEGIN {print $GPRT_TIME_MS/1000}")

    # Run Arkade
    echo "   Running Arkade..."
    ARKADE_TIME="N/A"
    if [ -f "$ARKADE_BIN" ] && [ -f "$ARKADE_COMB" ]; then
        shuf -n 10000 "$ARKADE_DATA" > /tmp/arkade_sample.txt 2>/dev/null || true
        
        RADIUS=$(python3 -c "
import numpy as np
from scipy.spatial import cKDTree
try:
    X = np.loadtxt('/tmp/arkade_sample.txt')
    tree = cKDTree(X)
    dists, _ = tree.query(X[:1000], k=5)
    print(np.percentile(dists[:, -1], 10))
except:
    print('0.1')
" 2>/dev/null || echo "0.1")

        if [[ "$RADIUS" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
            ARKADE_OUT=$($ARKADE_BIN $ARKADE_COMB $NPOINTS $NQUERIES $RADIUS 2>&1 || true)
            ARKADE_TIME=$(echo "$ARKADE_OUT" | grep -o -E "[0-9]+\.[0-9]+" | tail -n 1 || echo "N/A")
        fi
    fi

    echo "   Result -> Arkade: ${ARKADE_TIME}s | GPRT: ${GPRT_TIME_S}s"
    echo "$DS,$NPOINTS,$NQUERIES,$ARKADE_TIME,$GPRT_TIME_S" >> simple_results.csv
done

echo ""
echo "========================================="
echo " RESULTS (Seconds)"
echo "========================================="
column -s, -t < simple_results.csv
