#!/bin/bash
#
# run_centaloid.sh - OsiriX Plugin Helper Script
#
# This script is called by the CentiloidPlugin to run the Python
# Centiloid computation and generate DICOM output files.
#
# Usage: run_centaloid.sh <input_dicom_dir> <output_dir>
#

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input_dicom_dir> <output_dir>" >&2
    exit 1
fi

# Try to find Python 3
PYTHON=""
for p in /usr/local/bin/python3 /opt/homebrew/bin/python3 /usr/bin/python3; do
    if [ -x "$p" ]; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3 not found" >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Centiloid pipeline with DICOM output
exec "$PYTHON" -m centaloid.main run "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --dicom-sc \
    --dicom-sr \
    --verbose
