# Artifact for LEGOSim:A Unified Parallel Simulation Framework for Multi-chiplet Heterogeneous Integration

## Experiment Workflow

### 1. Run multi-chiplet system with different topologies and flit sizes
---
To run the multi-chiplet system with different topologies, you can use the provided script `run.sh` in the directory of each benchmark. This script will iterate through various topologies (mesh, meshll, NVL, star, torus) and flit sizes (2, 4), applying necessary modifications to the configuration files.

- bfs:
```bash
cd bfs_cuda
bash run.sh
```
- matmul:
```bash
cd matmul 
bash run.sh
```
- MLP:
```bash
cd mlp
bash run.sh
```
- Transformer:

As the transformer benchmark was built with libtorch library, libtorch 2.0.0+cpu and gcc/g++ 9.4.0 or higher is required. You can set the compiler in the `CMakeLists.txt` file as follows:
```bash
set(CMAKE_PREFIX_PATH "<LibTorch path>") 
# In CMakeLists.txt, change the compiler to gcc-9 and g++-9
set(CMAKE_C_COMPILER "<gcc-9.4.0 path>/bin/gcc")
set(CMAKE_CXX_COMPILER "<gcc-9.4.0 path>/bin/g++")
```
Then, compile the transformer to test the environment:
```bash
cd transformer
mkdir build
cd build
cmake ..
make -j4
```
To run the transformer benchmark with different topologies and flit sizes, use the following command:
```bash
cd ..
bash run.sh
```
Note: The MLP and Transformer benchmarks need several day to finish all the simulations. If you want to have a quick test, you can run bfs or matmul benchmarks at first.

### 2. Analyse computation or communication bottleneck of each chiplet
- Run:

```bash
python bottleneck_analyse.py --popnet_log ./<benchmark>/proc_r1_p2_t0 --cpu_node <cpu_node_id>
```
- Result:
```bash
# Example:
All deduplicated communication records:
...
Node Communication Statistics:
...
Node Computation and Communication Analysis:
Node    Total Comp Time Total Comm Time τ Value Bottleneck Type
----------------------------------------------------------------------
2       1847    1792022 0.0010306793108566747   Communication Bound
6       15709   3796598 0.004137651655508431    Communication Bound
7       51059   33      1547.2424242424242      Computation Bound
12      14316   3499685 0.004090653873134296    Communication Bound
13      51059   43      1187.4186046511627      Computation Bound
18      1153    900052  0.0012810370956344744   Communication Bound
Total computation time: 135143
Total communication time: 9988433
Average τ value: 0.013529950093272888

Computation Time Details for Each Node:
...
```

### 3. Run multi-chiplet system with different inter-chiplet communication protocols (PCIe, UCIe)
---
To run the multi-chiplet system with different inter-chiplet communication protocols, you can enter the `UCIe_PCIe` directory and execute the `run.sh` script. This script will iterate through the different protocols and apply necessary modifications to the configuration files.

```bash
cd UCIe_PCIe
bash run.sh
```

### 4. Run multi-chiplet system with different storage configurations (DDR5, HBM3)
---
To run the multi-chiplet system with different storage configurations, you can enter the `HBM_DDR` directory and execute the `run.sh` script. This script will iterate through the different storage configurations and apply necessary modifications to the configuration files.

```bash
cd HBM_DDR
bash run.sh
```

## Output Interpretation
The output of each benchmark will be stored in the `output` directory. The results will include the performance metrics for each topology and flit size combination at `result_{topology}_flit_{flit_size}.log`. You can analyze these results to compare the performance of different configurations. Besides, every simulation will generate a heat map which visualizes the inter-chiplet traffic distributions of each benchmark.

```bash
# Example output of result_{topology}_flit_{flit_size}.log
==== LegoSim Chiplet Simulator ====
Load benchmark configuration from ...
[info] **** Round 1 Phase 1 ****
[info] ...
[info] **** Round 1 Phase 2 ****
[info] ...
[info] **** Round 2 Phase 1 ****
[info] ...
[info] **** Round n Phase n ****
[info] ...
[info] Quit simulation because simulation cycle has converged.
[info] **** End of Simulation ****
[info] Benchmark elapses xxxxxx cycle.
[info] Simulation elapseds xx d xx h xxm xx s.
```
The expected results of above experiments are stored in the `output.csv` file.
