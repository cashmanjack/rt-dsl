#!/bin/bash
set -e

echo "========================================================="
echo " GPRT DSL vs Arkade vs RTBarnesHut Performance Suite"
echo "========================================================="

# Paths
GPRT_BIN="./target/release/gprt-sandbox"
ARKADE_DIR=~/Arkade
ARKADE_BIN=$ARKADE_DIR/build/sample02-withTrueknn
ARKADE_EST=$ARKADE_DIR/src/s02-withTrueknn/initial_estimate.py
DATASET_DIR=$ARKADE_DIR/datasets

# If you have the paper's C++ RTBarnesHut binary, set it here:
RTBH_BIN=~/RT_DSL/owl_barneshut/build/hostCode 

OUT_FILE="final_benchmark_results.csv"
echo "Dataset,N_Points,N_Queries,K,GPRT_GPU_Search_ms,GPRT_Total_EndToEnd_ms,Arkade_Search_ms,RTBarnesHut_Search_ms" > $OUT_FILE

# Datasets to benchmark (Add your synthetic CSVs here too if desired)
declare -a DATASETS=("3droad" "kitti" "porto")
K=5

echo "Building GPRT DSL in Release Mode..."
cargo build --release

for DS in "${DATASETS[@]}"; do
    echo "---------------------------------------------------------"
    echo "Benchmarking: $DS (k=$K)"
    echo "---------------------------------------------------------"
    
    # File paths
    GPRT_DATA="benchmarks/data/${DS}.csv"
    GPRT_QUERY="benchmarks/queries/${DS}_queries.csv"
    ARKADE_DATA=$DATASET_DIR/${DS}_data.txt
    ARKADE_COMB=$DATASET_DIR/${DS}_combined.txt
    
    if [ ! -f "$GPRT_DATA" ]; then
        echo "Warning: $GPRT_DATA not found. Skipping."
        continue
    fi

    NPOINTS=$(wc -l < "$GPRT_DATA")
    NQUERIES=$(wc -l < "$GPRT_QUERY")

    # ==========================================
    # 1. RUN GPRT DSL (Rust/OptiX)
    # ==========================================
    echo "[1/3] Running GPRT DSL..."
    # Capture both stdout and stderr to get the [GPRT_STATS] print
    GPRT_OUT=$($GPRT_BIN $GPRT_DATA $GPRT_QUERY $K 2>&1) 
    
    # Extract pure GPU search time from the macro's print statement
    GPRT_GPU_MS=$(echo "$GPRT_OUT" | grep -o -P '(?<=search_ms=)[0-9\.]+' || echo "N/A")
    # Extract total end-to-end time from the CSV output line
    GPRT_TOTAL_MS=$(echo "$GPRT_OUT" | tail -n 1 | cut -d',' -f4)

    # ==========================================
    # 2. RUN ARKADE (TrueKNN C++/OWL Baseline)
    # ==========================================
    echo "[2/3] Running Arkade..."
    ARKADE_TIME="N/A"
    if [ -f "$ARKADE_BIN" ] && [ -f "$ARKADE_COMB" ]; then
        # Get radius estimate
        EST_OUT=$(python3 $ARKADE_EST $ARKADE_DATA 1000 2>/dev/null || echo "")
        RADIUS=$(echo "$EST_OUT" | grep -o -E "[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?" | tail -n 1)
        
        if [ -n "$RADIUS" ]; then
            ARKADE_OUT=$($ARKADE_BIN $ARKADE_COMB $NPOINTS $NQUERIES $RADIUS 2>&1 || echo "")
            # Arkade prints search time as the last float
            ARKADE_TIME=$(echo "$ARKADE_OUT" | grep -o -E "[0-9]+\.[0-9]+" | tail -n 1)
            # Convert to ms if Arkade reports in seconds
            if [ -n "$ARKADE_TIME" ]; then
                ARKADE_TIME=$(awk "BEGIN {print $ARKADE_TIME * 1000}")
            fi
        fi
    else
        echo "   (Arkade binary or dataset not found, skipping)"
    fi

    # ==========================================
    # 3. RUN RTBARNESHUT (Paper's C++ Baseline)
    # ==========================================
    echo "[3/3] Running RTBarnesHut..."
    RTBH_TIME="N/A"
    if [ -f "$RTBH_BIN" ]; then
        # Adapt this command to match how the paper's hostCode accepts arguments
        # RTBH_OUT=$($RTBH_BIN $ARKADE_COMB $NPOINTS $NQUERIES 0.5 2>&1 || echo "")
        # RTBH_TIME=$(echo "$RTBH_OUT" | grep -o -P '(?<=Kernel Time: )[0-9\.]+' || echo "N/A")
        echo "   (Update RTBH_BIN command in script to match your C++ binary arguments)"
    else
        echo "   (RTBarnesHut binary not found, skipping)"
    fi

    # ==========================================
    # SAVE RESULTS
    # ==========================================
    echo "$DS,$NPOINTS,$NQUERIES,$K,$GPRT_GPU_MS,$GPRT_TOTAL_MS,$ARKADE_TIME,$RTBH_TIME" >> $OUT_FILE
done

echo ""
echo "========================================================="
echo " BENCHMARK COMPLETE (Times in Milliseconds)"
echo "========================================================="
column -s, -t < $OUT_FILE
