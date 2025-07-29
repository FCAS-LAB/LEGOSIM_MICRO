
For the detail user manual documents, please access [User Manual for LEGOSim ](./LEGOSim_Manual.pdf).

For the artifact document, please access [Artifact for LEGOSim](./artifact/README.md).

# Installation

## Download Repository and Setup Environment

**Necessary packages for installation**: zlib1g-dev, libbz2-dev, libboost-dev, libsqlite3-dev, xutils-dev, bison, flex, libgl1-mesa-dev, libglu1-mesa-dev, libboost-all-dev, CMake 3.26.3.

---
1. Clone the repository from GitHub.

    ```
    git clone https://github.com/FCAS-LAB/LEGOSIM_MICRO.git
    ```

    Enter the simulator root directory. The following examples assume execution from the simulator root directory.

2. Initialize and update submodules.

    ```
    git submodule init
    git submodule update
    ```

3. Run script to initialize environment variables.

    ```
    # In setup_env.sh, modify export PATH=/usr/local/cuda-11.3/bin:$PATH according to your CUDA version and path
    source setup_env.sh
    ```

    Success will show: setup_environment succeeded

4. Apply modifications to snipersim and gpgpu-sim code.

    ```
    ./apply_patch.sh
    ```

    See "Packaging and Applying Patches" section below for more details.

5. Compile and install snipersim. The new version provides automated compilation scripts, just run make.

    ```
    cd snipersim
    make -j4
    ```

6. Compile and install Gem5. Check Gem5 documentation for detailed installation guide. LEGOSim can run X86 and ARM architecture simulations:

    ```
    cd gem5
    scons build/X86/gem5.opt
    ```

    or

    ```
    cd gem5
    scons build/ARM/gem5.opt
    ```

7. Compile and install GPGPUSim. Prerequisites for GPGPUSim:

    1. GPGPUSim requires CUDA installation. The new version supports CUDA 4 to 11, see GPGPUSim README for details.
    2. GPGPUSim has compiler version requirements, GCC 7 recommended. For libraries like libtorch that need higher GCC versions, set during benchmark compilation.

    After configuring CUDA and compiler, run make.

    ```
    cd gpgpu-sim
    make -j4
    ```

8. Compile and install popnet

    ```
    cd popnet_chiplet
    mkdir build
    cd build
    cmake ..
    make -j4
    ```

9. Compile and install inter-chiplet communication program. interchiplet provides APIs and implementation code for inter-chiplet communication.

    ```
    cd interchiplet
    mkdir build
    cd build
    cmake ..
    make
    ```

    After compilation, find record_transfer and zmq_pro in interchiplet/bin, and libinterchiplet_app.a in interchiplet/lib.

    zmq_pro requires ZMQ environment. Usually skipped during cmake step.

# Verify Installation

After correct execution of above steps, use benchmark/matmul_test to verify environment setup.

1. Set simulator environment

    ```
    source setup_env.sh
    ```

2. Compile executable

    ```
    cd benchmark/matmul_test
    make
    ```

3. Execute the program. Example includes 4 processes: 1 CPU process and 3 GPU processes. Must execute in benchmark/matmul_test directory.

    ```
    make run
    ```

    After execution, find folders named proc_r{R}_p{P}_t{T} in benchmark/matmul_test, corresponding to thread T of phase P in round R.
    In folders you'll find:

    1. GPGPUSim temporary files and log file gpgpusim_X_X.log
    2. Sniper temporary files and log file sniper.log
    3. Popnet log file popnet.log

4. Clean executable and output files.

    ```
    make clean
    ```

# Packaging and Applying Patches

Since sniper and GPGPUSim are introduced as submodules, modifications aren't tracked through normal git workflow. The project provides patch.sh and apply_patch.sh scripts to manage modifications through patches.

patch.sh script generates patches:

```
./patch.sh
```

1. Packages modifications to snipersim.diff and gpgpu-sim.diff in interchiplet/patch. diff files are git tracked.
2. Also saves modified files in .changed_files folder maintaining directory structure, for reference if diff files have issues.

apply_patch.sh script applies patches:

```
./apply_patch.sh
```

1. Applies snipersim.diff, gpgpu-sim.diff, gem5.diff, popnet.diff and scale-sim.diff to snipersim, gpgpu-sim , gem5, popnet and scale-sim.
2. If application fails, refer to files in .changed_files for manual modification.

Note: Not recommended to directly copy from .changed_files to snipersim and gpgpu-sim folders. Snipersim and gpgpu-sim evolution might modify same files as chiplet simulator. Patches report conflicts, direct copying causes unpredictable errors.

# Adding Test Programs

Test programs are added to benchmark path, each with separate folder.

Recommended to organize like matmul and use similar Makefile, but not mandatory.

Running tests requires YAML configuration file.

## YAML Configuration File Format

```
# Phase 1 configuration.
phase1:
  # Process 0
  - cmd: "$BENCHMARK_ROOT/bin/matmul_cu"
    args: ["0", "1"]
    log: "gpgpusim.0.1.log"
    is_to_stdout: false
    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"
  # Process 1
  - cmd: "$BENCHMARK_ROOT/bin/matmul_cu"
    args: ["1", "0"]
    log: "gpgpusim.1.0.log"
    is_to_stdout: false
    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"
  ......

# Phase 2 configuration.
phase2:
  # Process 0
  - cmd: "$SIMULATOR_ROOT/popnet/popnet"
    args: ["-A", "36", "-c", "1", "-V", "3", "-B", "12", "-O", "12", "-F", "4", "-L", "1000", "-T", "1000000000", "-r", "1", "-I", "../bench.txt","-R", "4", "-G", "../topology/NVL_6_6_flit_4.gv","-R","4","-D", "../delayInfo.txt", "-P"]
    log: "popnet.log"
    is_to_stdout: false
```

First level YAML keywords:

- `phase1`: Configure first phase simulator processes
- `phase2`: Configure second phase simulator processes

Both keywords contain arrays, each item corresponding to concurrent simulator process. Both support multiple processes.

Process configuration keywords:

- `cmd`: Simulator command. String. Supports environment variables `$BENCHMARK_ROOT` and `$SIMULATOR_ROOT`
- `args`: Simulator parameters. String array. Supports environment variables `$BENCHMARK_ROOT` and `$SIMULATOR_ROOT`
- `log`: Log name. No relative/absolute paths
- `is_to_stdout`: Whether to redirect simulator stdout/stderr to interchiplet stdout
- `pre_copy`: Some simulators need extra files. String. Multiple files separated by spaces, enclosed in quotes

Relative paths in YAML use current path as base. Recommended to use environment variables for absolute paths.

- `$BENCHMARK_ROOT`: Test program path, determined by YAML location
- `$SIMULATOR_ROOT`: Simulator path, set by setup_env.sh

## Running InterChiplet

Main simulator program is InterChiplet. Execute in run path:

```
$SIMULATOR_ROOT/interchiplet/bin/interchiplet $BENCHMARK_ROOT/bench.yml
```

InterChiplet command format:

```
interchiplet <bench>.yml [--cwd <string>] [-t|--timeout <int>] [-e|--error <float>] [-h] [-w <int>] [-f <int>]
```

Command parameters:

- `<bench>.yml`: Test program configuration file
- `--cwd <string>`: Simulation execution path
- `-t <int>` and `--timeout <int>`: Simulation exit rounds. Ends regardless of convergence
- `-e <float>` and `--error <float>`: Simulation exit condition. Ends when error below this ratio
- `-w <int>`: Topology width, must match custom topology
- `-f <int>`: Flit size in 64-bit units, must match popnet configuration

