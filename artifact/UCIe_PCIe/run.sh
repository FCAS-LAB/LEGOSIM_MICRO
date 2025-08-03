#!/bin/bash

set -e  # 遇到错误就退出

# 测试配置列表
configs=(
    "8 mesh"
    "32 mesh"
)

# 运行测试函数
run_test() {
    local flit_size=$1
    local topology=$2
    local storage
    if [ "$flit_size" == "8" ]; then
        storage="PCIe"  
    fi

    if [ "$flit_size" == "32" ]; then
        storage="UCIe"  
    fi
    local result_file="result_${topology}_${storage}.log"
    
    echo "Testing: ${topology} with flit_size ${flit_size}"
    
    python modify.py --flit_size "${flit_size}" --topology "${topology}"
    make
    make run > "${result_file}" 2>&1
    python result.py --input "${result_file%.log}"
    make clean
    
    echo "Completed: ${result_file}"
}

# 主循环
echo "Starting batch tests..."

for config in "${configs[@]}"; do
    read -r flit_size topology <<< "$config"
    run_test "$flit_size" "$topology"
done

echo "All tests completed!"