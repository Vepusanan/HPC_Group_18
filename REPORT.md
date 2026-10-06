# Performance Comparison of CPU and CUDA Implementations of Conway’s Game of Life

**High Performance Computing — Group 18**  
**Experiment date:** 6 October 2026  
**Execution environments:** Local Apple M1 CPU and Google Colab NVIDIA GPU

> **Experimental status:** CPU measurements are complete. The CUDA implementation and execution notebook are prepared, but no current CUDA result file is available. Consequently, this report presents measured CPU results and the CUDA comparison methodology without claiming GPU timings, speedups or completed GPU validation.

## Abstract

This project implements Conway’s Game of Life using a sequential C++ CPU program and a CUDA C++ GPU program. Both use identical deterministic initialization, finite-grid boundary conditions and double buffering. The experiment evaluates five square grids, from 256 × 256 to 4096 × 4096, with exactly 100 generations per measured run and three repetitions per size. The CPU program was executed locally on an Apple M1 Mac. Its mean execution time increased from 22.711 ms for the smallest grid to 5,370.054 ms for the largest. The CUDA program is intended to execute on an NVIDIA GPU in Google Colab, with separate simulation and transfer-inclusive timing. A static dashboard loads the CPU JSON and accepts the CUDA JSON to calculate speedups and display the comparison. Since the two programs execute on different machines, eventual speedup values will describe the measured configurations rather than isolate the effect of GPU parallelization.

## 1. Introduction and objective

The objective is to compare the execution time and technical characteristics of CPU and CUDA implementations of the same cellular automaton. Game of Life is suitable for this experiment because each output cell depends only on a small neighbourhood in the previous generation. Cells within a generation can therefore be computed independently, while successive generations must remain ordered.

The project follows a simple workflow:

```text
Mac CPU benchmark → cpu_results.json → local dashboard
Colab CUDA benchmark → cuda_results.json → upload → comparison
```

The MacBook Air M1 is used for CPU execution, development and presentation. CUDA execution is assigned to Google Colab because the local machine cannot run the NVIDIA CUDA implementation. No backend, database or cloud deployment is required for the dashboard.

## 2. Conway’s Game of Life

Each cell stores either 0 (dead) or 1 (alive). Its eight surrounding positions form its neighbourhood. The next generation follows four rules:

1. A live cell with fewer than two live neighbours dies.
2. A live cell with two or three live neighbours survives.
3. A live cell with more than three live neighbours dies.
4. A dead cell with exactly three live neighbours becomes alive.

This is also described as **B3/S23**: birth with three neighbours, survival with two or three. Every cell reads the previous generation, so births and deaths occur simultaneously.

The experiment uses a finite square grid with a **dead exterior**. Positions outside the grid contribute zero to the neighbour count. Cells on the outermost rows and columns still evolve; there is no wrap-around.

Both programs generate their initial grids in row-major order with the same 32-bit linear congruential generator and seed 42. The high bit of each generated value supplies a cell state. Repeating a configuration therefore reproduces its initial grid.

## 3. CPU implementation

The CPU program, `cpu/game_of_life_cpu.cpp`, represents a logical two-dimensional array using one contiguous `std::vector<int>` allocation per buffer. Cell `(row, col)` is accessed at `row * N + col`. This arrangement avoids separately allocated rows and provides a straightforward layout for sequential traversal and comparison with CUDA.

The main functions are:

| Function | Purpose |
|---|---|
| `initializeGrid()` | Create the deterministic initial state |
| `countNeighbors()` | Sum neighbouring cells within the domain |
| `nextCellState()` | Apply the Game of Life rules |
| `updateGrid()` | Calculate one complete next generation |
| `runSimulation()` | Repeat updates and swap buffer pointers |

Two preallocated buffers hold the current and next grids. Nested loops traverse every cell, reading only the current buffer and writing only the next. After a generation, the program swaps pointers in constant time. Updating one buffer in place would be incorrect because later cells could read values already changed in that generation.

The implementation uses one host thread and no explicit CPU parallelism. The optimizing compiler may still apply instruction-level optimization or vectorization. CPU simulation time is measured with `std::chrono::steady_clock` around the update loop and pointer swaps. Initialization, allocation, final-grid export and printed output are outside this interval.

## 4. CUDA implementation

The baseline in `cuda/game_of_life_cuda.cu` assigns one valid CUDA thread to each output cell. Thread coordinates are calculated from the block index, block dimensions and thread index. Threads beyond the grid dimensions return without writing output.

The launch uses **16 × 16 threads per block**, or 256 threads. The number of blocks in each dimension is rounded upward to cover the domain. This is a practical baseline configuration, not a claim of optimal occupancy. A warp spans two 16-cell row segments, and neighbouring columns access adjacent addresses. Actual transaction efficiency would require profiling.

The host initializes the grid and allocates two device buffers. After an untimed warm-up, the original input is restored. During the measured simulation, both buffers remain in GPU memory. Each launch reads the current grid, writes the next grid and is followed by a host-side pointer swap. Launches in the same default stream preserve generation order. Only the initial input and final output are transferred during the measured total interval.

CUDA calls and kernel launches are checked for errors, and the stop event is synchronized before simulation timing is read. The implementation uses global memory; an optional shared-memory variant is not required for this experiment.

## 5. Experimental environment

The available CPU metadata comes directly from `results/cpu_results.json`.

| Item | CPU experiment | CUDA experiment |
|---|---|---|
| Location | Local Mac | Google Colab GPU runtime |
| Processor | Apple M1 | Pending actual allocation |
| Operating system | macOS 26.2, ARM64 | Pending runtime metadata |
| Compiler | Apple clang 17.0.0, clang-1700.4.4.1 | `nvcc`; version pending |
| Compilation flags | `-O3 -std=c++17` | Configured: `-O3 -std=c++17` |
| Recorded CPU run time | 6 October 2026, 15:01:22 Asia/Colombo | Pending |
| Execution status | Completed | Not yet measured |

**The CPU benchmark is not executed in the same environment as the GPU benchmark.** The Colab workflow also compiles a CPU reference, but that program is used for correctness checks. Its timing is not the CPU dataset used by the dashboard.

## 6. Benchmark methodology

### 6.1 Workload and repetition

Each measured run starts from the same initial state for its grid size and executes exactly **100 generations**. Three separate runs are collected for each of the following sizes:

| Grid | Cells | Cell updates across 100 generations |
|---|---:|---:|
| 256 × 256 | 65,536 | 6,553,600 |
| 512 × 512 | 262,144 | 26,214,400 |
| 1024 × 1024 | 1,048,576 | 104,857,600 |
| 2048 × 2048 | 4,194,304 | 419,430,400 |
| 4096 × 4096 | 16,777,216 | 1,677,721,600 |

These sizes increase the cell count by a factor of four at each step and allow scaling to be observed over a substantial range. Two four-byte integer grids at the largest size occupy approximately 128 MiB, excluding other allocations.

The runner retains all timing samples and reports their arithmetic mean and sample standard deviation. Slow samples are not discarded. The standard deviation describes observed variation; it is not a confidence interval.

### 6.2 Timing scopes

| Metric | Measurement method | Included work |
|---|---|---|
| CPU simulation | Monotonic host clock | 100 updates and pointer swaps |
| CUDA simulation | CUDA events | Device timeline interval around 100 launches |
| CUDA total | Monotonic host clock | H2D copy, simulation, synchronization and D2H copy |

CUDA simulation timing can include gaps between launches. CUDA total excludes device allocation, context setup, warm-up and export, so it is not the duration of the entire executable. The difference between total and simulation timing includes transfer and host/synchronization effects; it does not isolate pure transfer latency.

The CUDA process performs one untimed warm-up kernel before restoring the initial state. CPU timings retain ordinary initial cache effects. This methodological difference must be considered alongside the hardware differences.

### 6.3 Speedup definition

For each compatible grid size:

```text
Simulation speedup = mean CPU simulation time / mean CUDA simulation time
Total speedup      = mean CPU simulation time / mean CUDA total time
```

A ratio above one means CUDA is faster under the selected timing definition. Ratios are calculated from the mean times, not by averaging per-run ratios. No speedup can currently be reported because the CUDA measurements are pending.

## 7. Correctness verification

CPU tests have passed for all rule combinations, a stable block, a blinker oscillator, small and boundary-sensitive grids, and 100 successive generations against an independent reference. The reference distributes each live cell’s contribution to its neighbours instead of using the production neighbour-counting loop. Address and undefined-behaviour sanitizer checks also passed during development.

For each measured CPU size, repeated runs produced the same final-grid SHA-256 fingerprint. This establishes repeatability for those runs, alongside the algorithm tests.

The Colab runner is configured to test tiny grids, partial blocks and odd/even generation counts. It then compares each measured CUDA output byte-for-byte against the CPU reference for that size. Failed validation prevents a new dataset from being exported. **These GPU checks remain to be executed.**

The dashboard checks compatible experiment settings and matches CPU and CUDA records by grid size and final-grid SHA-256. A mismatch suppresses that pair’s speedup. Matching fingerprints provide strong cross-machine consistency evidence, but are distinct from an exact browser-side comparison of the full grids.

## 8. Measured results

### 8.1 CPU execution time

All values below are milliseconds for 100 generations, rounded to three decimal places. Unrounded measurements are retained in the JSON file.

| Grid | Run 1 | Run 2 | Run 3 | Mean | Sample standard deviation |
|---|---:|---:|---:|---:|---:|
| 256 × 256 | 29.835 | 19.318 | 18.982 | 22.711 | 6.171 |
| 512 × 512 | 75.730 | 76.571 | 75.020 | 75.773 | 0.776 |
| 1024 × 1024 | 299.884 | 301.914 | 301.502 | 301.100 | 1.073 |
| 2048 × 2048 | 1,303.447 | 1,227.509 | 1,236.406 | 1,255.787 | 41.514 |
| 4096 × 4096 | 6,376.190 | 4,878.801 | 4,855.170 | 5,370.054 | 871.420 |

### 8.2 CPU scaling

| Increase in grid dimension | Cell-count factor | Mean CPU time factor |
|---|---:|---:|
| 256 → 512 | 4.00× | 3.34× |
| 512 → 1024 | 4.00× | 3.97× |
| 1024 → 2048 | 4.00× | 4.17× |
| 2048 → 4096 | 4.00× | 4.28× |

Mean CPU time grows at every tested size. Beyond the smallest pair, multiplying the cell count by four increases execution time by approximately four, consistent with the algorithm’s linear work per cell at a fixed generation count. This observation does not establish a specific hardware bottleneck.

Variation is not uniform. The 256 × 256 measurements have a standard deviation of about 27.2% of their mean, while the 4096 × 4096 measurements have about 16.2%. The first measured sample is noticeably slower than the other two at both sizes. The saved data cannot identify whether scheduling, cache behaviour, clock changes or another factor caused this difference. All samples are retained; additional repetitions would improve the assessment of stability.

### 8.3 CUDA comparison status

| Grid | CPU mean (ms) | CUDA simulation (ms) | CUDA total (ms) | Speedup |
|---|---:|---|---|---|
| 256 × 256 | 22.711 | Pending | Pending | Not calculated |
| 512 × 512 | 75.773 | Pending | Pending | Not calculated |
| 1024 × 1024 | 301.100 | Pending | Pending | Not calculated |
| 2048 × 2048 | 1,255.787 | Pending | Pending | Not calculated |
| 4096 × 4096 | 5,370.054 | Pending | Pending | Not calculated |

The present evidence cannot determine the highest GPU speedup, a crossover size or whether GPU overhead dominates smaller workloads. After a validated CUDA run, this table and its interpretation should be replaced with the dashboard’s comparison export. The dashboard uses a descriptive 2× threshold for a substantial advantage; that threshold is not a statistical significance test.

## 9. Technical characteristics and performance choices

| Area | CPU implementation | CUDA implementation |
|---|---|---|
| Execution | One host thread traverses cells | Threads organized into blocks update cells concurrently |
| Memory | Contiguous host arrays and hardware caches | Contiguous device arrays and device caches |
| Synchronization | Sequential loop order | Ordered launches and final synchronization |
| Data movement | Pointer swaps between host buffers | Pointer swaps on device-buffer addresses; initial/final transfers |
| Implementation effort | Simpler allocation and debugging | Device allocation, launch geometry and error handling |
| Work and storage | O(iterations × N²) work, O(N²) storage | Same total work and storage orders; parallel scheduling |
| Expected tradeoff | Avoids GPU transfer/launch costs | Can exploit sufficient parallel work; benefit must be measured |

The implemented performance choices are contiguous storage, preallocation, constant-time buffer swaps, compiler optimization and device-resident simulation. These choices remove avoidable allocation and copying. Their individual speedup contributions have not been measured through controlled before/after experiments. No claim is made that the selected block size is optimal or that memory bandwidth is saturated.

## 10. Visualization and exported data

The dashboard uses HTML, CSS, JavaScript and local SVG charts. It loads `results/cpu_results.json` automatically and accepts a CUDA JSON upload without manual entry of values. It shows mean times, variability, grid sizes, iterations, simulation speedup and transfer-inclusive speedup. Unsupported comparisons remain empty rather than displaying example measurements.

Insights are derived from compatible measurements: the actual maximum speedup, time growth between tested sizes, whether speedup is monotonic, and the relationship between CUDA simulation and total time. JSON exports retain environment metadata, raw timing samples and result fingerprints; the dashboard can export a combined CSV.

A separate 48 × 48 interactive JavaScript simulation provides Start, Pause, Step, Reset and Randomize controls. It demonstrates the rules only. Its animation is not part of the C++ or CUDA benchmark.

## 11. Limitations

- The Mac CPU and Colab GPU use different hardware, operating environments and compilers. Speedup describes the complete measured configurations.
- The CPU baseline is single-threaded, not an optimized multicore implementation. Results should not be generalized to all CPU approaches.
- Only three samples and one deterministic initial state are used per size. This limits statistical precision and workload coverage.
- CPU and GPU warm-up treatment differs. Neither background load nor thermal and clock behaviour is fully controlled.
- Timing excludes allocation and startup. Application-level execution time can therefore exceed the reported intervals.
- Dead-exterior boundaries differ from an infinite or periodic Game of Life universe.
- GPU correctness and performance conclusions are pending the actual Colab run.

## 12. Conclusion

The CPU experiment demonstrates a working, tested implementation with reproducible final states across five grid sizes and exactly 100 generations per run. Mean CPU time increases from approximately 22.7 ms to 5.37 seconds as the workload grows. The larger-size results exhibit roughly proportional growth with cell count, while some configurations show substantial run-to-run variation.

The CUDA baseline, validation workflow and upload-based comparison interface are prepared. Completion of the empirical CPU-versus-CUDA comparison requires running the notebook, saving its validated results and updating this report with the measured GPU timings and speedups. Until then, the project supports conclusions about CPU scaling, but not GPU superiority.

## Appendix — Reproduction and completion

From the project root on the Mac:

```bash
python3 run_benchmarks.py
python3 -m http.server 8080 --bind 127.0.0.1
```

Open `http://127.0.0.1:8080/visualizer/`. In Google Colab, upload `notebooks/Conway_Game_of_Life_HPC_Colab.ipynb`, enable a GPU runtime, run all cells and download `cuda_results.json`. Upload that file to the dashboard and save the comparison CSV.

For final submission, retain both original JSON datasets, the executed notebook, source code, this report, chart screenshots and the presentation video. Complete the GPU environment fields and results table only from the current validated CUDA export. Rerunning the CPU benchmark replaces its result file; update the report if those measurements change.

## References and project evidence

1. [Conway’s Game of Life — rules and background](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).
2. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
3. [Recorded CPU measurements and environment](results/cpu_results.json).
4. [CPU implementation](cpu/game_of_life_cpu.cpp), [CUDA implementation](cuda/game_of_life_cuda.cu), [benchmark runner](run_benchmarks.py), and [CPU correctness tests](tests/test_cpu.cpp).
