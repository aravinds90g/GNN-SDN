#!/bin/bash

# Batch Processing Script for Network Flow Data
# Processes all reduced_data_*.csv files in the current directory

echo "========================================"
echo "Network Flow Data Batch Preprocessing"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi

# Activate virtual environment
source venv/bin/activate

# Process each file
for i in 1 2 3 4; do
    input_file="reduced_data_$i.csv"
    output_file="preprocessed_data_$i.csv"
    
    if [ -f "$input_file" ]; then
        echo ""
        echo "Processing $input_file..."
        echo "----------------------------------------"
        
        ./venv/bin/python preprocess_network_data_optimized.py \
            "$input_file" \
            -o "$output_file" \
            --normalize
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed $input_file"
        else
            echo "❌ Failed to process $input_file"
        fi
    else
        echo "⚠️  File not found: $input_file"
    fi
done

echo ""
echo "========================================"
echo "Batch processing complete!"
echo "========================================"
echo ""
echo "Output files:"
ls -lh preprocessed_data_*.csv 2>/dev/null || echo "No output files generated"
