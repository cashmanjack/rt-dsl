#!/bin/bash
DATASET=$1
QUERY=$2
K=$3

if [ -z "$DATASET" ]; then
    echo "Usage: ./sweep_params.sh <data.csv> <queries.csv> <k>"
    exit 1
fi

echo "Sweeping Parameters for $(basename $DATASET)..."
echo "Percentile,Mult,Search_ms,Total_Neighbors" > sweep_results.csv

# Test different percentiles (0.10 = sparse, 0.90 = clustered)
for P in 0.10 0.50 0.90 0.95 0.99 1.00; do
    # Test different multipliers (1.5 = slow growth, 3.0 = fast growth)
    for M in 1.5 2.0 3.0; do
        echo -n "Testing P=$P, M=$M ... "
        
        # Run binary, capture stdout
        OUT=$(./target/release/gprt-sandbox $DATASET $QUERY $K $P $M 2>/dev/null)
        
        # Extract stats
        TIME=$(echo "$OUT" | grep "search_ms=" | grep -o -P '(?<=search_ms=)[0-9\.]+')
        NEIGHBORS=$(echo "$OUT" | tail -n 1 | cut -d',' -f5)
        
        echo "Done! ($TIME ms)"
        echo "$P,$M,$TIME,$NEIGHBORS" >> sweep_results.csv
    done
done

echo ""
echo "=== SWEEP RESULTS ==="
column -s, -t < sweep_results.csv
