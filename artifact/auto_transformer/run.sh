#!/bin/bash

set -e  # 遇到错误就退出

# 测试配置列表
configs=(
    "2 mesh"
    "4 mesh"
    "2 meshll"
    "4 meshll"
    "2 torus"
    "4 torus"
    "2 star"
    "4 star"
    "2 NVL"
    "4 NVL"
)


run_test() {
    local flit_size=$1
    local topology=$2
    local result_file="result_${topology}_flit_${flit_size}.log"
    local bottleneck_file="bottleneck_${topology}_flit_${flit_size}.log"
    echo "Testing: ${topology} with flit_size ${flit_size}"
    
    flit_size=$((flit_size * 100))
    echo "Flit size adjusted to: ${flit_size}"
    python modify.py --flit_size "${flit_size}" --topology "${topology}"
    cd build
    cmake ..
    make
    make run > "${result_file}" 2>&1
    cd ..
    echo "Generate heatmap for ${topology} with flit_size ${flit_size}"
    python heatMap.py --topology "${topology}" --flit_size "${flit_size}"
    python bottleneck_analyse.py --popnet_log ./build/proc_r1_p2_t0/popnet_0.log > "${bottleneck_file}" 2>&1
    cd build
    make clean_target
    cd ..
    
    echo "Completed: ${result_file}"
}


echo "Starting batch tests..."

for config in "${configs[@]}"; do
    read -r flit_size topology <<< "$config"
    run_test "$flit_size" "$topology"
done

echo "All tests completed!"