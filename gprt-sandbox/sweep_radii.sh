#!/bin/bash
# sweep_radii.sh

DATASETS=("3droad" "porto")
# 2^-6, 2^-5, 2^-4, 2^-3 (current), 2^-2, 2^-1, 2^0
MULTIPLIERS=(0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0)

echo "Multiplier,Dataset,Time_ms,Total_Neighbors" > radius_sweep.csv

for MULT in "${MULTIPLIERS[@]}"; do
    echo "========================================="
    echo "Testing Radius Multiplier: $MULT"
    echo "========================================="
    
    # 1. Update the macro using sed
    sed -i -E "s/__chosen_dist \* 1\.1 \* [0-9.]+;/__chosen_dist * 1.1 * $MULT;/g" ../gprt-macros/src/lib.rs
    
    # 2. Rebuild silently
    cargo build --release -q
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
    
    # 3. Run benchmarks
    for DS in "${DATASETS[@]}"; do
        DATA_FILE="benchmarks/data/${DS}.csv"
        QUERY_FILE="benchmarks/queries/${DS}_queries.csv"
        
        # Run, hide stderr (the round logs), capture stdout (the CSV line)
        OUTPUT=$(./target/release/gprt-sandbox "$DATA_FILE" "$QUERY_FILE" 5 2>/dev/null)
        
        TIME_MS=$(echo "$OUTPUT" | cut -d',' -f4)
        NEIGHBORS=$(echo "$OUTPUT" | cut -d',' -f5)
        
        echo "$MULT,$DS,$TIME_MS,$NEIGHBORS" >> radius_sweep.csv
        echo "  [$DS] Time: ${TIME_MS}ms | Neighbors: $NEIGHBORS"
    done
done

echo ""
echo "Sweep complete! Results saved to radius_sweep.csv"
cat radius_sweep.csv
