#!/bin/bash
set -e

PROFILER_BIN="$HOME/RT_DSL/target/release/gprt-profiler"
SANDBOX_BIN="$HOME/RT_DSL/target/release/gprt-sandbox"
DATA_DIR="gprt-sandbox/benchmarks/data"
QUERY_DIR="gprt-sandbox/benchmarks/queries"

DATASETS=("3droad" "kitti" "3diono" "porto")
K=5

echo "Dataset,GPRT_NativeTuned_s" > native_results.csv

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo " Native Rust Profiling: $DS"
    echo "========================================="
    
    DATA=$DATA_DIR/$DS.csv
    QUERY=$QUERY_DIR/${DS}_queries.csv
    
    if [ ! -f "$DATA" ]; then continue; fi

    # 1. Run the Native Rust Profiler (The Funnel)
    # It will test subsets, verify accuracy, and write the optimal JSON
    echo "   Running gprt-profiler (Rust Native)..."
    $PROFILER_BIN $DATA $QUERY $K
    
    # 2. Run the Sandbox using the newly generated gprt_wisdom.json
    echo "   Running full dataset with tuned wisdom..."
    OUT=$($SANDBOX_BIN $DATA $QUERY $K 2>&1)
    TIME_MS=$(echo "$OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+' || echo "0")
    TIME_S=$(awk "BEGIN {print $TIME_MS/1000}")
    
    echo "   Result -> $DS: ${TIME_S}s"
    echo "$DS,$TIME_S" >> native_results.csv
done

echo ""
echo "========================================="
echo " FINAL NATIVE RUST RESULTS (Seconds)"
echo "========================================="
column -s, -t < native_results.csv
