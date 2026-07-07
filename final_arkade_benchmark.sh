#!/bin/bash
set -e

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OPTIX_PATH=/home/min/a/cashman3/optix_sdk

ARKADE_BIN=~/Arkade/build/sample02-withTrueknn
DATASET_DIR=~/Arkade/datasets

GPRT_BIN="$HOME/RT_DSL/target/release/gprt-sandbox"
GPRT_DATA_DIR="gprt-sandbox/benchmarks/data"
GPRT_QUERY_DIR="gprt-sandbox/benchmarks/queries"

DATASETS=("3droad" "kitti" "3diono" "porto")
K=5

echo "Dataset,Arkade_s,GPRT_NativeTuned_s,GPRT_Fixed_s" > final_paper_benchmark.csv

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Benchmarking $DS (k=$K)"
    echo "========================================="
    
    GPRT_DATA=$GPRT_DATA_DIR/$DS.csv
    GPRT_QUERY=$GPRT_QUERY_DIR/${DS}_queries.csv
    ARKADE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    if [ ! -f "$GPRT_DATA" ]; then continue; fi

    NPOINTS=$(wc -l < "$GPRT_DATA")
    NQUERIES=$(wc -l < "$GPRT_QUERY")

    if [ "$DS" == "3droad" ]; then RADIUS="3.5414e-05"; fi
    if [ "$DS" == "kitti" ]; then RADIUS="0.0155"; fi
    if [ "$DS" == "3diono" ]; then RADIUS="0.0680"; fi
    if [ "$DS" == "porto" ]; then RADIUS="9.99e-06"; fi

    ARKADE_TIME="N/A"
    if [ -f "$ARKADE_BIN" ] && [ -f "$ARKADE_COMB" ]; then
        echo "   Running Arkade (Hardcoded Radius: $RADIUS)..."
        sleep 1 
        ARKADE_OUT=$($ARKADE_BIN $ARKADE_COMB $NPOINTS $NQUERIES $RADIUS 2>&1 || true)
        ARKADE_TIME=$(echo "$ARKADE_OUT" | grep -o -E "^[0-9]+\.[0-9]+$" | tail -n 1)
        if [ -z "$ARKADE_TIME" ]; then ARKADE_TIME="CRASH"; fi
        
        echo "   [DEBUG ARKADE] Raw Output Tail:"
        echo "$ARKADE_OUT" | tail -n 3
    fi

    # 2. GPRT Native Auto-Tuned
    rm -f gprt_wisdom.json
    echo "   Running GPRT (Native Full-Dataset Autotuner)..."
    cd gprt-sandbox
    GPRT_AUTO_OUT=$(../target/release/gprt-sandbox benchmarks/data/${DS}.csv benchmarks/queries/${DS}_queries.csv $K tune 2>&1)
    cd ..
    
    echo "   [DEBUG GPRT] Autotuner Selection & Final Time:"
    echo "$GPRT_AUTO_OUT" | grep -E "(Winner Selected|search_ms=)" | tail -n 2
    
    GPRT_AUTO_MS=$(echo "$GPRT_AUTO_OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' | tail -n 1 || echo "0")
    GPRT_AUTO_S=$(awk "BEGIN {printf \"%.6f\", $GPRT_AUTO_MS/1000.0}")

    # 3. GPRT Fixed Baseline
    rm -f gprt_wisdom.json
    echo "   Running GPRT (Fixed P=0.1, M=3.0)..."
    cd gprt-sandbox
    GPRT_FIXED_OUT=$(../target/release/gprt-sandbox benchmarks/data/${DS}.csv benchmarks/queries/${DS}_queries.csv $K 0.10 3.0 2>&1)
    cd ..
    GPRT_FIXED_MS=$(echo "$GPRT_FIXED_OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' | tail -n 1 || echo "0")
    GPRT_FIXED_S=$(awk "BEGIN {printf \"%.6f\", $GPRT_FIXED_MS/1000.0}")

    echo "   -> Arkade: ${ARKADE_TIME}s | GPRT (Native): ${GPRT_AUTO_S}s | GPRT (Fixed): ${GPRT_FIXED_S}s"
    echo "$DS,$ARKADE_TIME,$GPRT_AUTO_S,$GPRT_FIXED_S" >> final_paper_benchmark.csv
done

echo ""
echo "========================================="
echo " FINAL HEAD-TO-HEAD RESULTS (Seconds)"
echo "========================================="
column -s, -t < final_paper_benchmark.csv
