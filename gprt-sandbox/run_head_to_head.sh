#!/bin/bash

ARKADE_DIR=~/Arkade
ARKADE_BIN=$ARKADE_DIR/build/sample02-withTrueknn
ARKADE_EST=$ARKADE_DIR/src/s02-withTrueknn/initial_estimate.py
DATASET_DIR=$ARKADE_DIR/datasets

GPRT_BIN=~/RT_DSL/gprt-sandbox/target/release/gprt-sandbox
GPRT_DATA_DIR=~/RT_DSL/gprt-sandbox/benchmarks/data
GPRT_QUERY_DIR=~/RT_DSL/gprt-sandbox/benchmarks/queries

DATASETS=("3droad" "kitti" "3diono" "porto")
K=5

echo "Dataset,Arkade_Time_s,GPRT_Time_s" > head_to_head_results.csv

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Benchmarking $DS (k=$K)"
    echo "========================================="
    
    ARKADE_FILE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_FILE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    NPOINTS=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $2}')
    NQUERIES=$(grep "^$DS " $DATASET_DIR/counts.txt | awk '{print $3}')
    
    echo "Data points: $NPOINTS, Queries: $NQUERIES"
    
    echo "1. Getting radius estimate for Arkade..."
    EST_OUT=$(python3 $ARKADE_EST $ARKADE_FILE_DATA 1000 2>/dev/null)
    RADIUS=$(echo "$EST_OUT" | grep -o -E "[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?" | tail -n 1)
    
    if [ -z "$RADIUS" ]; then
        echo "   ❌ Failed to get radius estimate. Skipping Arkade."
        ARKADE_TIME="N/A"
    else
        echo "   Radius estimate: $RADIUS"
        echo "2. Running Arkade TrueKNN (C++/OWL, k=$K)..."
        ARKADE_OUT=$($ARKADE_BIN $ARKADE_FILE_COMB $NPOINTS $NQUERIES $RADIUS 2>&1)
        echo "$ARKADE_OUT" | tail -n 6
        
        # Arkade prints: radius, KN, npoints, nsearchpoints, build_time, search_time
        # The search time is the very last float printed.
        ARKADE_TIME=$(echo "$ARKADE_OUT" | grep -o -E "[0-9]+\.[0-9]+" | tail -n 1)
    fi
    
    echo "3. Running GPRT DSL (Rust/OptiX, k=$K)..."
    GPRT_DATA=$GPRT_DATA_DIR/$DS.csv
    GPRT_QUERY=$GPRT_QUERY_DIR/${DS}_queries.csv
    GPRT_OUT=$($GPRT_BIN $GPRT_DATA $GPRT_QUERY $K 2>/dev/null)
    echo "$GPRT_OUT"
    
    # GPRT prints CSV: path,queries,k,time_ms,neighbors
    GPRT_TIME_MS=$(echo "$GPRT_OUT" | cut -d',' -f4)
    
    # Convert GPRT time from ms to seconds
    if [ -n "$GPRT_TIME_MS" ]; then
        GPRT_TIME_S=$(awk "BEGIN {print $GPRT_TIME_MS/1000}")
    else
        GPRT_TIME_S="N/A"
    fi
    
    echo "$DS,$ARKADE_TIME,$GPRT_TIME_S" >> head_to_head_results.csv
done

echo ""
echo "========================================="
echo "Head-to-Head Comparison Results (Seconds, k=5)"
echo "========================================="
column -s, -t < head_to_head_results.csv
