# Performance Comparison of CPU and CUDA Implementations of Conway’s Game of Life

**High Performance Computing — Group 18**  
**Experiment date:** 7 October 2026  
**Platforms:** Apple M1 Mac CPU and NVIDIA Tesla T4 GPU in Google Colab

## Abstract

This project compares sequential CPU and parallel CUDA implementations of Conway’s Game of Life. Both implementations use the same deterministic initialization, boundary conditions and update rules. Five square grids, ranging from 256 × 256 to 4096 × 4096, are simulated for exactly 100 iterations per measured run, with three repetitions per size. The CPU implementation runs locally on an Apple M1 Mac, while the CUDA implementation runs on a Tesla T4 GPU in Google Colab. Mean CPU time increases from 23.511 ms to 4,846.510 ms, and mean CUDA simulation time increases from 0.347 ms to 89.437 ms. CUDA is faster at every tested size under both simulation-only and transfer-inclusive timing. The maximum simulation speedup is 93.353× at 512 × 512; the maximum transfer-inclusive speedup is 46.871× at 2048 × 2048. Speedup is not monotonic with grid size. All five final-grid fingerprints match, and the CUDA dataset records successful exact CPU-reference validation. The results compare these complete hardware and software configurations rather than isolate GPU parallelism as the sole cause of the improvement.

## 1. Introduction and objectives

Conway’s Game of Life is a useful example of a stencil computation: each cell’s next state depends on a small neighbourhood in the preceding generation. The output cells within one generation can be calculated independently, making the problem suitable for GPU parallelization. Successive generations must nevertheless execute in order.

The objectives are to implement the same simulation on CPU and CUDA, verify consistent outputs, measure execution time over several grid sizes, and compare both performance and implementation characteristics. A lightweight web application presents exported measurements alongside an interactive demonstration.

The experiment follows this workflow:

```text
Local CPU benchmark → CPU JSON → visualizer
Colab CUDA benchmark → CUDA JSON → upload → comparison
```

The CPU benchmark runs on the Mac. The GPU benchmark runs in Colab because the Mac does not provide an NVIDIA CUDA execution environment. Colab also runs a CPU reference for correctness checks; those reference timings are not used in the performance comparison.

## 2. Simulation rules and data representation

Each cell is represented by 0 (dead) or 1 (alive) and has at most eight neighbours. The rules are:

1. A live cell with fewer than two live neighbours dies.
2. A live cell with two or three live neighbours survives.
3. A live cell with more than three live neighbours dies.
4. A dead cell with exactly three live neighbours becomes alive.

These rules are commonly written as **B3/S23**: birth with three neighbours and survival with two or three. Updates are simultaneous: every output cell reads the previous generation, not partially updated values. The experiment uses a finite square grid with a **dead exterior**. Out-of-domain positions contribute zero; edge cells inside the grid still evolve, and there is no wrap-around.

Both implementations initialize cells in row-major order using the same 32-bit linear congruential generator with seed 42. The most significant bit of each generated value determines a cell’s initial state. Identical dimensions therefore produce identical initial grids.

## 3. CPU implementation

The C++ implementation in `cpu/game_of_life_cpu.cpp` stores each logical two-dimensional grid in a contiguous `std::vector<int>`. The element at `(row, col)` is accessed using `row * N + col`. This avoids separate row allocations and supports sequential memory access.

| Function | Responsibility |
|---|---|
| `initializeGrid()` | Generate the initial cell states |
| `countNeighbors()` | Count valid live neighbours |
| `nextCellState()` | Apply the four rules |
| `updateGrid()` | Compute one complete generation |
| `runSimulation()` | Repeat updates and swap buffer pointers |

Two grids are allocated before simulation: `currentGrid` and `nextGrid`. Nested loops read from the current grid and write each output cell into the next grid. Their pointers are then swapped in constant time. This preserves simultaneous-update semantics without copying the entire grid between generations.

The baseline uses one host thread and no explicit multithreading. Compiler optimizations may still exploit instruction-level parallelism or vectorization. Compilation uses `-O3 -std=c++17`.

## 4. CUDA implementation

The CUDA implementation in `cuda/game_of_life_cuda.cu` assigns one valid thread to each output cell. Row and column coordinates are derived from block and thread indices. A bounds check prevents threads outside the domain from accessing invalid cells.

Blocks contain **16 × 16 threads**, giving 256 threads or eight warps per block. The launch grid rounds the required number of blocks upward in each dimension. This configuration maps naturally to the two-dimensional domain. Neighbouring columns use adjacent memory addresses, although a warp spans two 16-cell row segments and neighbour accesses can be unaligned. The configuration is a reasonable baseline, not a demonstrated optimum.

The host allocates two device buffers and initializes the input. An untimed kernel warm-up is followed by restoration of the original input. For the measured workload, the input is copied to the GPU once, 100 kernels are launched in the same default stream, and the final grid is copied back once. Between launches, the host swaps device-buffer pointers. Stream ordering preserves dependencies between generations without requiring a host synchronization after each launch.

CUDA API calls and kernel launches are checked for errors. Synchronizing the stop event ensures that the simulation has completed before its timing is read. This implementation uses global memory; shared-memory tiling is not part of the measured experiment.

## 5. Experimental environment

The following information is recorded in the two result files.

| Item | CPU experiment | CUDA experiment |
|---|---|---|
| Execution location | Local MacBook Air | Google Colab GPU runtime |
| Processor | Apple M1 | NVIDIA Tesla T4 |
| Operating environment | macOS 26.2, ARM64 | Linux 6.6.122+, x86_64, glibc 2.39 |
| Compiler | Apple clang 17.0.0, clang-1700.4.4.1 | NVIDIA CUDA compiler 13.0, V13.0.88 |
| Compiler flags | `-O3 -std=c++17` | `-O3 -std=c++17` |
| Dataset timestamp, Asia/Colombo | 7 October 2026, 20:00:31 | 7 October 2026, 20:03:15 |

The timestamps come from `createdAt`, which the runner sets before the benchmark sweep; they are not completion times. The GPU driver version and the Colab CPU-reference hardware/compiler details are not recorded in these exports.

**The performance measurements come from different machines.** Consequently, the speedup incorporates differences in processor architecture, memory systems, compiler behaviour and runtime conditions, as well as the execution model. Identical workload settings support a useful comparison, but do not make this a controlled same-machine experiment.

## 6. Benchmark methodology

### 6.1 Workloads and repetitions

Every measured run uses exactly 100 iterations. Each size is executed three times in separate processes, starting from the same seed and initial state. The runner tests sizes in ascending order and retains all samples.

| Grid | Cells | Cell updates per measured run |
|---|---:|---:|
| 256 × 256 | 65,536 | 6,553,600 |
| 512 × 512 | 262,144 | 26,214,400 |
| 1024 × 1024 | 1,048,576 | 104,857,600 |
| 2048 × 2048 | 4,194,304 | 419,430,400 |
| 4096 × 4096 | 16,777,216 | 1,677,721,600 |

Doubling the grid width quadruples the number of cells. At 4096 × 4096, two grids of four-byte integers occupy 128 MiB, excluding other host allocations and runtime overhead. Both implementations perform O(I × N²) total work for I iterations and use O(N²) grid storage. Parallel execution changes elapsed time, not the number of cell updates required by this algorithm.

### 6.2 Timing definitions

| Metric | Clock | Included work |
|---|---|---|
| CPU simulation | `std::chrono::steady_clock` | 100 CPU updates and pointer swaps |
| CUDA simulation | CUDA events | Device timeline interval around 100 kernel launches |
| CUDA total | `std::chrono::steady_clock` | Initial host-to-device copy, simulation, synchronization and final device-to-host copy |

All measurements exclude compilation, allocation, initialization, validation, printing and file export. CUDA total also excludes context setup and warm-up; it therefore measures the transfer-inclusive simulation, not complete application startup and execution.

CUDA event timing may include device timeline gaps between launches. Total-minus-simulation time includes transfers and host/synchronization overhead and is not a separate measurement of transfer latency. CUDA performs a warm-up kernel in each process; CPU measurements do not include an equivalent explicit warm-up. CPU caches may still have been affected by initialization. This asymmetry is retained as a limitation rather than described as a fully controlled warm-up policy.

### 6.3 Statistics and speedup

For the three samples, the arithmetic mean is the reported execution time and the sample standard deviation uses a denominator of two. Standard deviation describes observed variation, not a confidence interval. Slow samples are not removed.

```text
Simulation speedup = mean CPU simulation time / mean CUDA simulation time
Total speedup      = mean CPU simulation time / mean CUDA total time
```

A ratio above one indicates lower CUDA time for the selected scope. The ratios use unrounded mean times, not averages of individual run ratios. Displaying three decimal places is a formatting choice and does not imply that the measurements are accurate to that precision.

## 7. Correctness and data verification

Correctness is checked at three levels:

1. **CPU algorithm tests.** The test suite covers all rule combinations, a stable block, a blinker, boundary cases, and 100 generations against an independently structured reference that distributes live-cell contributions. These tests passed during development, including address and undefined-behaviour sanitizer checks.
2. **Exact CUDA comparison.** The Colab runner checks small grids and partial blocks with odd and even iteration counts. For every benchmark size, it compares each measured final CUDA grid byte-for-byte against the C++ CPU reference. All five CUDA records report `exact-cpu-reference`. A mismatch prevents a new dataset from being exported.
3. **Cross-machine consistency.** The local CPU and CUDA datasets have matching experiment settings, final-grid SHA-256 fingerprints and living-cell counts for every size. Fingerprint matching is strong evidence of consistent final results, while the exact comparison is performed separately in Colab.

The dataset review also recomputed the means and simulation standard deviations, checked CUDA total means against their samples, and verified that both source hashes match the current implementation files. This report uses the supplied exports; the report review did not rerun the benchmarks. The recorded final states do not establish equality at every intermediate generation of every large-grid run.

## 8. Results

### 8.1 Execution times and speedups

CPU and CUDA simulation times are **mean ± sample standard deviation**, in milliseconds, across three runs. CUDA total is the mean transfer-inclusive time. The table preserves the visualizer’s measurements and formatting. All rows use 100 iterations and compatible results with matching final-grid fingerprints. Ratios calculated from the rounded displayed times can differ slightly from the exported speedups.

| Grid / cells | Iterations | CPU ms | CUDA ms | CUDA total ms | Speedup | Total speedup | Result check |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 256 × 256 / 65,536 | 100 | 23.511 ± 5.861 | 0.347 ± 0.011 | 0.505 | 67.765× | 46.531× | SHA-256 match |
| 512 × 512 / 262,144 | 100 | 77.065 ± 0.416 | 0.826 ± 0.004 | 1.98 | 93.353× | 38.916× | SHA-256 match |
| 1024 × 1024 / 1,048,576 | 100 | 299.852 ± 1.068 | 4.798 ± 0.5 | 6.681 | 62.494× | 44.884× | SHA-256 match |
| 2048 × 2048 / 4,194,304 | 100 | 1,205.305 ± 1.164 | 19.206 ± 0.003 | 25.715 | 62.758× | 46.871× | SHA-256 match |
| 4096 × 4096 / 16,777,216 | 100 | 4,846.51 ± 43.354 | 89.437 ± 12.01 | 118.552 | 54.189× | 40.881× | SHA-256 match |

CUDA has lower mean simulation and total times at all five sizes. The largest simulation speedup is **93.353× at 512 × 512**. The largest transfer-inclusive speedup is **46.871× at 2048 × 2048**. Neither maximum occurs at the largest grid.

For the 512 × 512 case, using additional digits from the saved means:

```text
Simulation speedup = 77.064653 / 0.825514667 ≈ 93.353×
Total speedup      = 77.064653 / 1.980282333 ≈ 38.916×
```

The CUDA simulation therefore takes approximately one ninety-third of the CPU time for this case. Including transfers and synchronization reduces the measured advantage to approximately 38.916×.

### 8.2 Final-state agreement

| Grid | CPU final living cells | CUDA final living cells | Final-grid SHA-256 |
|---|---:|---:|---|
| 256 × 256 | 5,877 | 5,877 | Match |
| 512 × 512 | 23,852 | 23,852 | Match |
| 1024 × 1024 | 99,296 | 99,296 | Match |
| 2048 × 2048 | 390,059 | 390,059 | Match |
| 4096 × 4096 | 1,584,269 | 1,584,269 | Match |

The complete fingerprints remain in the original JSON files. The matching living-cell counts are supplementary diagnostics; equal counts alone would not establish equal grids.

## 9. Performance analysis

### 9.1 Scaling with problem size

| Increase in grid dimension | Cell-count factor | CPU mean-time factor | CUDA simulation factor | CUDA total factor |
|---|---:|---:|---:|---:|
| 256 → 512 | 4.00× | 3.28× | 2.38× | 3.92× |
| 512 → 1024 | 4.00× | 3.89× | 5.81× | 3.37× |
| 1024 → 2048 | 4.00× | 4.02× | 4.00× | 3.85× |
| 2048 → 4096 | 4.00× | 4.02× | 4.66× | 4.61× |

After the smallest pair, CPU execution time grows approximately fourfold whenever the number of cells quadruples. This is consistent with the fixed amount of work performed per cell at 100 iterations. CUDA simulation time grows less uniformly: the first increase is below fourfold, while the 512-to-1024 and 2048-to-4096 increases exceed fourfold.

Across increasing sizes, simulation speedup is **67.765× → 93.353× → 62.494× → 62.758× → 54.189×**. GPU performance remains advantageous, but relative speedup does not increase monotonically. Timing data alone cannot attribute these changes to caching, memory bandwidth, scheduling or utilization; such explanations would require additional profiling.

The smallest tested grid already has a substantial measured advantage for CUDA. All tested sizes exceed the dashboard’s descriptive 2× threshold under both timing definitions. This threshold is not a statistical significance test, and the experiment does not identify the crossover point because smaller workloads were not tested.

### 9.2 Effect of transfers and host overhead

The additional measured interval is CUDA total minus CUDA simulation. The percentage below is relative to CUDA total time.

| Grid | Total minus simulation (ms) | Share of CUDA total time |
|---|---:|---:|
| 256 × 256 | 0.158 | 31.33% |
| 512 × 512 | 1.155 | 58.31% |
| 1024 × 1024 | 1.883 | 28.18% |
| 2048 × 2048 | 6.510 | 25.31% |
| 4096 × 4096 | 29.115 | 24.56% |

Including this additional work reduces speedup at every size, but CUDA total remains below CPU simulation time throughout. The highest proportional difference occurs at 512 × 512, where a slow total-time sample affects the mean. The data therefore do not support a simple claim that overhead always decreases with grid size. These differences combine several costs and should not be labelled pure PCIe or kernel-launch measurements.

### 9.3 Run-to-run variation

| Grid | CPU SD (ms) | CUDA simulation SD (ms) | CUDA total SD (ms) |
|---|---:|---:|---:|
| 256 × 256 | 5.861 | 0.011 | 0.021 |
| 512 × 512 | 0.416 | 0.004 | 0.948 |
| 1024 × 1024 | 1.068 | 0.500 | 0.833 |
| 2048 × 2048 | 1.164 | 0.003 | 0.440 |
| 4096 × 4096 | 43.354 | 12.010 | 12.676 |

CUDA total standard deviations are derived from the saved `totalSamplesMs` arrays. The CPU’s largest relative standard deviation is **24.93% at 256 × 256**; it is below 1% at the other four sizes. CUDA simulation relative standard deviation reaches **10.42% at 1024 × 1024** and **13.43% at 4096 × 4096**.

At 512 × 512, CUDA total samples are approximately 1.431, 3.075 and 1.435 ms. The slower second sample raises the total mean despite relatively stable simulation times. All samples are retained. With only three repetitions, the experiment cannot provide a precise characterization of timing distributions or establish the cause of slower runs.

## 10. Technical comparison and optimizations

| Area | CPU implementation | CUDA implementation |
|---|---|---|
| Execution model | One host thread with nested loops | Threads grouped into blocks and scheduled on the GPU |
| Memory | Contiguous host arrays and CPU caches | Device global arrays and GPU caches |
| Generation ordering | Sequential control flow | Ordered launches in the same stream |
| Buffer management | Two preallocated grids and pointer swaps | Two resident device grids and pointer swaps |
| Communication | No discrete GPU transfers | Input/output transfers across the host-device boundary |
| Complexity | Simpler control, allocation and debugging | Additional allocation, launch, synchronization and error handling |
| Suitable conditions | Useful when GPU overhead or availability is limiting | Can exploit sufficient parallel work; measured advantage depends on hardware and workload |

The performance-oriented choices are:

- **Contiguous storage:** supports row-major traversal and straightforward GPU transfers.
- **Preallocation and pointer swaps:** avoid allocation and full-grid copies inside generation loops.
- **Compiler optimization:** both builds use `-O3`; equal flag names do not imply identical compiler transformations.
- **Device-resident simulation:** avoids host-device transfers between generations.
- **Two-dimensional thread blocks:** map cell coordinates directly to GPU threads and encourage adjacent-column accesses.
- **CUDA compiler hints:** neighbour loops request unrolling, and restricted pointer declarations describe non-aliasing input/output buffers.

These choices reduce avoidable work, but the experiment does not measure each optimization in isolation. No claim is made that 16 × 16 blocks are optimal, shared-memory tiling would be faster, or the kernel saturates device memory bandwidth.

## 11. Visualization and reproducibility

The web application uses HTML, CSS, JavaScript and SVG charts without a backend or external chart dependency. It loads the local CPU JSON and accepts a CUDA JSON upload, then pairs compatible grid sizes and result fingerprints. The interface presents execution times, speedups, timing variation and insights calculated from the loaded data. A mismatched pair is excluded from speedup calculations.

The execution-time chart uses a logarithmic vertical axis to show both implementations over their wide timing range. The speedup chart distinguishes simulation and transfer-inclusive ratios. A separate 48 × 48 JavaScript simulation demonstrates Start, Pause, Step, Reset and Randomize; its animation time is excluded from the benchmarks.

The JSON files retain sample timings, environment metadata and fingerprints. The dashboard can export the comparison as CSV. Keeping these files alongside the source and executed notebook makes the calculations traceable.

## 12. Limitations

1. **Different systems:** the Mac and Colab differ in processor, memory, compiler and runtime conditions. The ratios describe the measured configurations.
2. **Single-threaded CPU baseline:** the results do not represent an optimized multicore CPU implementation or all possible CPU algorithms.
3. **Limited sampling:** three repetitions, one seed and ascending size order provide limited coverage of variability and workload behaviour.
4. **Warm-up and environmental effects:** warm-up differs between implementations, and background load, clock changes and thermal conditions are not controlled or profiled.
5. **Timing scope:** initialization, allocation and startup are excluded, so complete application runtime can exceed the reported values.
6. **Finite workload:** only five sizes and dead-exterior boundaries are tested. Smaller-grid crossover behaviour and other boundary conditions are outside the experiment.
7. **Recorded evidence:** the report review verifies supplied data and source consistency rather than repeating GPU execution. Driver details, Colab reference CPU metadata and profiling counters are unavailable in the exports.

## 13. Conclusion

The project implements and evaluates the same Game of Life workload on a local Apple M1 CPU and a Tesla T4 GPU in Google Colab. Both datasets cover five grid sizes, 100 iterations per run and three repetitions per size. Their final-grid fingerprints and living-cell counts agree, and the CUDA export records successful exact CPU-reference checks.

CUDA has lower measured execution time at every tested size, including transfers. Simulation speedup peaks at **93.353× for 512 × 512**, while transfer-inclusive speedup peaks at **46.871× for 2048 × 2048**. At the largest grid, CPU mean time is **4,846.510 ms**, CUDA simulation time is **89.437 ms**, and CUDA total time is **118.552 ms**, corresponding to **54.189×** and **40.881×** speedups.

The evidence supports a substantial advantage for this CUDA implementation on the tested GPU. It also shows that increasing grid size does not guarantee increasing speedup, and that transfers and timing variability matter when interpreting results. The conclusions apply to the measured configurations and timing scopes, rather than establishing a universal GPU-versus-CPU performance ratio.

## Appendix A. Reproducing the experiment

From the project root on the Mac, run the CPU benchmark:

```bash
python3 run_benchmarks.py
```

Start the local dashboard server:

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open `http://127.0.0.1:8080/visualizer/`. Upload `notebooks/Conway_Game_of_Life_HPC_Colab.ipynb` to Colab, select a GPU runtime and run all cells. Download `cuda_results.json`, retain a copy in `results/`, and upload it through the dashboard to view the comparison.

The report uses the datasets dated 7 October 2026. New benchmark runs can produce different times and replace result files; retain the originals associated with the submitted report. Submission evidence should include both JSON datasets, the executed notebook, source code, comparison charts and the presentation video.

## Appendix B. Individual timing samples

All values are milliseconds for 100 iterations, rounded to three decimal places.

| Grid | CPU runs 1 / 2 / 3 | CUDA simulation runs 1 / 2 / 3 | CUDA total runs 1 / 2 / 3 |
|---|---|---|---|
| 256 × 256 | 30.270 / 20.439 / 19.825 | 0.341 / 0.360 / 0.340 | 0.498 / 0.529 / 0.489 |
| 512 × 512 | 76.910 / 76.748 / 77.536 | 0.821 / 0.829 / 0.826 | 1.431 / 3.075 / 1.435 |
| 1024 × 1024 | 300.726 / 300.167 / 298.662 | 5.144 / 5.025 / 4.225 | 7.410 / 6.860 / 5.772 |
| 2048 × 2048 | 1,204.125 / 1,206.451 / 1,205.340 | 19.207 / 19.208 / 19.202 | 25.420 / 26.221 / 25.506 |
| 4096 × 4096 | 4,822.055 / 4,820.908 / 4,896.566 | 97.347 / 95.346 / 75.618 | 126.567 / 125.150 / 103.937 |

## References and project evidence

1. [Conway’s Game of Life — rules and background](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).
2. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
3. [CPU measurements and environment](results/cpu_results.json) and [CUDA measurements and environment](results/cuda_results.json).
4. [CPU source](cpu/game_of_life_cpu.cpp), [CUDA source](cuda/game_of_life_cuda.cu), [benchmark runner](run_benchmarks.py) and [correctness tests](tests/test_cpu.cpp).
