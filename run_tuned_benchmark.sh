#!/bin/bash
set -e

# Array of all datasets in the benchmark suite
DATASETS=("3droad" "kitti" "3diono" "porto")

echo "========================================================="
# Professional, objective logging
echo " STARTING FULL DATASET AUTOTUNING PIPELINE"
echo "========================================================="
echo " This will sweep, verify correctness, and optimize each dataset."
echo ""

for DS in "${DATASETS[@]}"; do
    echo "---------------------------------------------------------"
    echo " Tuning $DS..."
    echo "---------------------------------------------------------"
    python3 autotune.py "$DS"
    echo ""
done

echo "========================================================="
echo " RUNNING FINAL HEAD-TO-HEAD BENCHMARK AGAINST ARKADE"
echo "========================================================="
# Execute the benchmark harness with our newly generated wisdom files
./final_arkade_benchmark.sh
