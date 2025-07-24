#!/bin/bash

set -e

configs=(
    "DDR"
    "HBM"
)

run_test() {
    local storage_type=$1
    local result_file="result_${storage_type}.log"
    echo "Testing: ${storage_type}"
    python modify.py --type "${storage_type}"
    make
    make run > "${result_file}" 2>&1
    make clean
    echo "Completed: ${result_file}"
}

echo "Starting DDR and HBM tests..."

for config in "${configs[@]}"; do
    run_test "$config"
done