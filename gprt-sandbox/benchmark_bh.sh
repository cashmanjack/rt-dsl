#!/bin/bash

DATASETS=("$HOME/synthetic10M.csv" "$HOME/synthetic25M.csv" "$HOME/synthetic50M.csv")
THETA=0.5

DSL_BIN="./target/release/barneshut"
OWL_BIN="$HOME/OWLRayTracing/build/rtbarneshut"

echo "Dataset,DSL_Total_ms,OWL_Preprocess_s,OWL_Kernel_s,OWL_Total_s" > bh_head_to_head.csv

for DS in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Benchmarking $(basename $DS) (Theta=$THETA)"
    echo "========================================="
    
    # 1. Run DSL
    echo "[1/2] Running GPRT DSL..."
    DSL_OUT=$($DSL_BIN "$DS" $THETA)
    DSL_TIME=$(echo "$DSL_OUT" | grep "Execution Time" | awk '{print $4}')
    echo "DSL End-to-End Time: $DSL_TIME ms"
    
    # 2. Run OWL Artifact
    echo "[2/2] Running Original RT-BarnesHut (OWL)..."
    # Pass 'csv' as the first argument, then the file path
    OWL_OUT=$($OWL_BIN csv "$DS" 2>&1) 
    
    # Parse the specific timers from the OWL output
    OWL_PRE=$(echo "$OWL_OUT" | grep "Preprocessing Time:" | grep -o -E "[0-9]+\.[0-9]+" | head -n 1)
    OWL_KERNEL=$(echo "$OWL_OUT" | grep "RT Cores Force Calculations time:" | grep -o -E "[0-9]+\.[0-9]+" | head -n 1)
    OWL_TOTAL=$(echo "$OWL_OUT" | grep "Execution time:" | grep -o -E "[0-9]+\.[0-9]+" | head -n 1)
    
    echo "OWL Preprocess: ${OWL_PRE}s | OWL Kernel: ${OWL_KERNEL}s | OWL Total: ${OWL_TOTAL}s"
    
    echo "$(basename $DS),$DSL_TIME,$OWL_PRE,$OWL_KERNEL,$OWL_TOTAL" >> bh_head_to_head.csv
done

echo ""
echo "=== Head-to-Head Results ==="
column -s, -t < bh_head_to_head.csv
