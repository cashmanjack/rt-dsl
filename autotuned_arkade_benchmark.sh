#!/bin/bash
set -e

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OPTIX_PATH=/home/min/a/cashman3/optix_sdk

ARKADE_DIR=~/Arkade
ARKADE_BIN=$ARKADE_DIR/build/sample02-withTrueknn
DATASET_DIR=$ARKADE_DIR/datasets

GPRT_BIN="$HOME/RT_DSL/target/release/gprt-sandbox"
GPRT_DATA_DIR="gprt-sandbox/benchmarks/data"
GPRT_QUERY_DIR="gprt-sandbox/benchmarks/queries"

DATASETS=("3droad" "kitti" "3diono" "porto")
K=5
PERCENTILES=(0.001 0.005 0.01 0.02 0.05 0.10 0.50)

echo "Dataset,Arkade_Autotuned_s,GPRT_Autotuned_s" > autotuned_arkade_benchmark.csv

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Autotuning Arkade & GPRT: $DS (k=$K)"
    echo "========================================="
    
    GPRT_DATA=$GPRT_DATA_DIR/$DS.csv
    GPRT_QUERY=$GPRT_QUERY_DIR/${DS}_queries.csv
    ARKADE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    if [ ! -f "$GPRT_DATA" ]; then continue; fi

    NPOINTS=$(wc -l < "$GPRT_DATA")
    NQUERIES=$(wc -l < "$GPRT_QUERY")

    BEST_ARKADE_TIME=999999.0
    BEST_ARKADE_RADIUS=0.1

    if [ -f "$ARKADE_BIN" ] && [ -f "$ARKADE_COMB" ]; then
        echo "   Sweeping Tiny Percentiles for Arkade..."
        for P in "${PERCENTILES[@]}"; do
            RADIUS=$(python3 -c "
import numpy as np
from scipy.spatial import cKDTree
X = np.loadtxt('$ARKADE_DATA')
np.random.seed(42)
size = min(10000, len(X))
sample = X[np.random.choice(len(X), size, replace=False)]
tree = cKDTree(sample)
dists, _ = tree.query(sample, k=6)
non_zero = dists[:, 5][dists[:, 5] > 1e-6]
if len(non_zero) > 0: print(np.percentile(non_zero, $P * 100))
else: print(0.1)
" 2>/dev/null || echo "0.1")
            
            echo -n "      P=$P (Radius=$RADIUS) -> "
            ARKADE_OUT=$(timeout 120 $ARKADE_BIN $ARKADE_COMB $NPOINTS $NQUERIES $RADIUS 2>&1 || true)
            TIME_S=$(echo "$ARKADE_OUT" | grep -o -E "[0-9]+\.[0-9]+" | tail -n 1)
            if [ -z "$TIME_S" ]; then TIME_S="999999.0"; fi
            echo "${TIME_S}s"
            
            IS_FASTER=$(awk "BEGIN {print ($TIME_S < $BEST_ARKADE_TIME) ? 1 : 0}")
            if [ "$IS_FASTER" -eq 1 ]; then
                BEST_ARKADE_TIME=$TIME_S
                BEST_ARKADE_RADIUS=$RADIUS
            fi
        done
        echo "   -> Best Arkade: Radius=$BEST_ARKADE_RADIUS (${BEST_ARKADE_TIME}s)"
    else
        BEST_ARKADE_TIME="N/A"
    fi

    # GPRT Native Auto-Tuned
    echo "   Running GPRT Native Autotuner..."
    rm -f gprt_wisdom.json
    $GPRT_BIN $GPRT_DATA $GPRT_QUERY $K tune 2>&1 | grep -E "(TUNE|VALID|REJECT|Winner)" || true
    
    echo "   Running GPRT Final Execution..."
    GPRT_AUTO_OUT=$($GPRT_BIN $GPRT_DATA $GPRT_QUERY $K 2>&1 || true)
    GPRT_AUTO_MS=$(echo "$GPRT_AUTO_OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' | tail -n 1)
    if [ -z "$GPRT_AUTO_MS" ]; then GPRT_AUTO_MS="0"; fi
    GPRT_AUTO_S=$(awk "BEGIN {printf \"%.6f\", $GPRT_AUTO_MS/1000.0}")

    echo "   -> Arkade (Autotuned): ${BEST_ARKADE_TIME}s | GPRT (Autotuned): ${GPRT_AUTO_S}s"
    echo "$DS,$BEST_ARKADE_TIME,$GPRT_AUTO_S" >> autotuned_arkade_benchmark.csv
done

echo ""
echo "========================================="
echo " FINAL AUTOTUNED HEAD-TO-HEAD (Seconds)"
echo "========================================="
column -s, -t < autotuned_arkade_benchmark.csv
