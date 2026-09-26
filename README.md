# Conway's Game of Life: CPU and CUDA

A small high-performance computing comparison of Conway's Game of Life. One program runs the simulation sequentially on the CPU. The other runs the same simulation on an NVIDIA GPU with CUDA. Both programs start from the same grid and should finish with the same grid.

## 1. Project objective

The objective is to compare performance, not to draw the simulation.

The CPU program is the baseline. The CUDA program does the same work with one GPU thread per cell. For each grid size, the report quantity is:

```text
speedup = CPU execution time / CUDA execution time
```

A speedup of 4 means the GPU run finished in a quarter of the CPU time. A speedup below 1 means the CPU run finished first. That can happen on small grids, because launching a GPU kernel has a cost and a small grid does not contain enough parallel work to hide it.

## 2. Conway's Game of Life

Each cell is alive (`1`) or dead (`0`). Every generation, each cell looks at its eight neighbours and follows these rules:

- A live cell with fewer than 2 live neighbours dies.
- A live cell with 2 or 3 live neighbours survives.
- A live cell with more than 3 live neighbours dies.
- A dead cell with exactly 3 live neighbours becomes alive.
- Every other dead cell stays dead.

Cells outside the grid are treated as dead. This is a zero boundary, not a wrap-around boundary.

A generation must be computed into a second grid. If a cell were overwritten in the grid currently being read, later neighbour counts in that same generation would see a mixture of old and new values.

The default problem size is a `1024 x 1024` grid for `100` generations. The assignment also uses `256`, `512`, `2048`, and `4096`.

## 3. CPU implementation

`cpu/game_of_life_cpu.cpp` is ordinary C++. It stores two grids, `currentGrid` and `nextGrid`, as contiguous row-major arrays. The cell at row `r` and column `c` is the element at index `r * N + c`.

The storage is one block of memory per grid. A `vector<vector<int>>` would store each row in a separate allocation. That would add pointer chasing and would make the CPU baseline look slower for a reason that has nothing to do with the Game of Life itself.

Each generation does five things:

1. Visit every cell.
2. Count the eight neighbours that lie inside the grid.
3. Apply the rules.
4. Write the new value into `nextGrid`.
5. Swap `currentGrid` and `nextGrid`.

The swap exchanges the two buffers. It does not copy the cells. After 100 swaps, `currentGrid` is generation 100.

Only that loop is timed, using `std::chrono::steady_clock`. Building the random grid, counting the final living cells, computing the checksum, and printing are outside the clock.

The random grid uses the fixed seed `42` and a short generator written directly in the source. `rand()` and `std::uniform_int_distribution` are not used, because their sequences can differ between C++ libraries. The CPU file and the CUDA file contain the same generator, so they build the same initial grid on the lab machine.

## 4. CUDA implementation

`cuda/game_of_life_cuda.cu` is the global-memory version. It does not use shared memory. The first goal is a correct parallel result that matches the CPU program.

What the program does:

1. Build the initial grid on the CPU with the same function and seed as the CPU program.
2. Allocate two device arrays, `d_current` and `d_next`, with `cudaMalloc`.
3. Copy the initial grid to the GPU once.
4. Launch one kernel per generation. Each thread computes one cell.
5. Swap the device pointers. The cells stay on the GPU.
6. After all generations, copy the final grid back once.

The launch shape is:

```cpp
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (N + 15) / 16);
```

A thread finds its cell with:

```text
column = blockIdx.x * blockDim.x + threadIdx.x
row    = blockIdx.y * blockDim.y + threadIdx.y
```

Threads whose row or column is outside `N` return immediately. That matters when `N` is not a multiple of 16.

Within one kernel, threads only read `d_current` and each thread writes its own cell of `d_next`. They do not need `__syncthreads()`. The next generation is a new kernel launch on the same stream, and that launch runs only after the previous kernel has finished. Generations stay in order. Cells inside one generation run together.

The grids remain in device memory for the whole simulation. Copying the grid back to the CPU after every generation would move far more data than the arithmetic requires, so the program does not do that.

Columns are stored next to each other. Threads with consecutive `threadIdx.x` handle consecutive columns, so those global-memory loads are coalesced. The block is 16 columns wide, which is the configuration required here.

GPU time is measured with CUDA events around the generation loop only. The events are recorded on the same stream as the kernels. The start event is queued after the initial copy, so the recorded interval begins when that copy has finished. The final copy happens after the stop event has completed, so it is not included. `cudaMalloc`, random initialisation, and printing are also outside the interval.

`cudaEventElapsedTime` reports milliseconds. The printed CUDA time includes kernel execution and the short gaps between launches. Those gaps are part of running 100 iterations, and they are the reason a small grid may show little or no speedup.

## 5. Project structure

```text
.
├── cpu/
│   └── game_of_life_cpu.cpp
├── cuda/
│   └── game_of_life_cuda.cu
├── results/
│   └── results.csv
├── README.md
└── .gitignore
```

Compile from this folder. The executables are created here and are listed in `.gitignore`.

## 6. Requirements

- Windows PC with an NVIDIA CUDA-capable GPU
- A C++ compiler for the CPU program: MinGW `g++`, or Microsoft Visual C++ (`cl`)
- The NVIDIA CUDA Toolkit, which provides `nvcc`
- No extra libraries, GUI, or Python packages

Open a terminal where the compiler is on `PATH`. For `cl`, use the **x64 Native Tools Command Prompt** from Visual Studio. For `nvcc`, use the **Developer Command Prompt** or another prompt opened after the CUDA Toolkit is installed. `nvcc -V` should print a version.

## 7. How to compile the CPU version on Windows

From this folder, with MinGW or another `g++`:

```bat
g++ -O2 -std=c++17 cpu/game_of_life_cpu.cpp -o game_of_life_cpu.exe
```

With Microsoft Visual C++:

```bat
cl /O2 /EHsc /std:c++17 cpu\game_of_life_cpu.cpp /Fe:game_of_life_cpu.exe
```

`-O2` and `/O2` turn on optimisation. The CPU time in the report should come from an optimised build. An unoptimised build is not a fair baseline.

## 8. How to compile the CUDA version on Windows

From the same folder:

```bat
nvcc -O2 cuda/game_of_life_cuda.cu -o game_of_life_cuda.exe
```

If that `nvcc` reports that the GPU architecture must be set, add the architecture of the lab GPU. CUDA 11.7 and newer accept:

```bat
nvcc -O2 -arch=native cuda/game_of_life_cuda.cu -o game_of_life_cuda.exe
```

`native` means the GPU in the machine used to compile. If the lab toolkit is older, use the compute capability from the GPU properties instead, for example `-arch=sm_75` or `-arch=sm_86`.

## 9. How to run both

Defaults are `N = 1024` and `100` iterations:

```bat
game_of_life_cpu.exe
game_of_life_cuda.exe
```

Or pass the grid size and the iteration count:

```bat
game_of_life_cpu.exe 1024 100
game_of_life_cuda.exe 1024 100
```

CPU output:

```text
Implementation: CPU
Grid size: 1024 x 1024
Iterations: 100
Execution time (ms): ...
Living cells: ...
Checksum: ...
```

CUDA output adds the GPU name and the block size:

```text
Implementation: CUDA
GPU name: ...
Grid size: 1024 x 1024
Iterations: 100
Block size: 16 x 16
Execution time (ms): ...
Living cells: ...
Checksum: ...
```

Use the same two numbers for both programs whenever you compare them.

## 10. How to benchmark

Run both programs for each grid size. The assignment asks for about 100 iterations, so keep the second argument at `100`.

```bat
game_of_life_cpu.exe 256 100
game_of_life_cuda.exe 256 100

game_of_life_cpu.exe 512 100
game_of_life_cuda.exe 512 100

game_of_life_cpu.exe 1024 100
game_of_life_cuda.exe 1024 100

game_of_life_cpu.exe 2048 100
game_of_life_cuda.exe 2048 100

game_of_life_cpu.exe 4096 100
game_of_life_cuda.exe 4096 100
```

Close other heavy programs first. The `4096` CPU run visits about 16 million cells, 100 times, and can take a few minutes.

For each size, calculate:

```text
speedup = CPU time in ms / CUDA time in ms
```

Write the printed times into `results/results.csv`. The speedup column is that division. Example, using placeholder times only to show the arithmetic: if the CPU time were `1000` ms and the CUDA time were `200` ms, the speedup would be `5`. Replace those placeholders with the numbers printed on the lab PC. Do not put invented times in the report.

Larger grids give the GPU more cells to update per launch, so speedup often grows with `N`. A `256 x 256` grid has far fewer cells and the kernel-launch cost is a larger share of the measured time. It is a valid result if that size is only slightly faster, or slower, than the CPU.

The work per generation is proportional to `N x N`. Every cell is visited, including dead cells, because any cell might be born. The runtime should grow much faster from `2048` to `4096` than from `256` to `512`, roughly with the number of cells, which quadruples when `N` doubles.

## 11. How correctness is validated

Both programs must agree on:

- grid size
- seed (`42`)
- initialisation order (row by row)
- the four Game of Life rules
- dead cells outside the border
- the number of generations

After the run, each program prints:

- the number of living cells in the final grid
- a checksum of the final grid

The checksum starts at `0`. For every cell in row-major order it does `checksum = checksum * 1315423911 + cell`. Changing any cell changes the number. If the living-cell counts match and the checksums match, the final grids match.

Compare those two lines before calculating speedup. A faster CUDA time is only meaningful when those values are the same. If they differ, the GPU run is not a correct acceleration of this CPU program.

The programs do not print the grid. A `4096 x 4096` grid is too large to compare by eye, which is why the checksum exists.

For seed `42` and `100` iterations, a correct run prints these values. Check a small size first. If the CUDA living-cell count or checksum differs, fix that before collecting timing results.

| Grid size | Living cells | Checksum |
| --- | ---: | ---: |
| 256 x 256 | 5877 | 2188031159639976069 |
| 512 x 512 | 23852 | 3838351066650152210 |
| 1024 x 1024 | 99296 | 14789994132222743192 |
| 2048 x 2048 | 390059 | 17074688330164608745 |
| 4096 x 4096 | 1584269 | 15231916768216565527 |

A quick check that also fits in a few seconds is `64 20`. Both programs should print `571` living cells and checksum `2642065318239818179`.

## 12. What to collect for the report

Fill `results/results.csv` from the console output:

| Column | Where it comes from |
| --- | --- |
| `grid_size` | The `N` you passed |
| `iterations` | `100` |
| `block_size` | `16x16`, printed by the CUDA program |
| `cpu_time_ms` | CPU line `Execution time (ms)` |
| `cuda_time_ms` | CUDA line `Execution time (ms)` |
| `speedup` | `cpu_time_ms / cuda_time_ms` |
| `living_cells` | Must be the same number from both programs |
| `checksum` | Must be the same number from both programs |
| `gpu_name` | CUDA line `GPU name` |

Also keep a screenshot or a pasted copy of one CPU printout and the matching CUDA printout, ideally for `1024` and for the largest size you run. State that both used seed `42`, dead borders, and 100 generations.

In the discussion, cover:

- The two programs produce the same living-cell count and checksum.
- Speedup for each grid size, including any size where it is near 1 or below 1.
- Why a larger grid can use the GPU more effectively: thousands of cells are updated at the same time, and the two grids stay on the device.
- Why generations themselves are not parallel: generation `k + 1` depends on generation `k`. Parallelism is across cells inside one generation.
- The measured CUDA time excludes allocation and the two full-grid copies, and includes the 100 kernel launches.

## Presentation outline

About three minutes:

1. **Rules (20 seconds).** Alive or dead, eight neighbours, the survival and birth rules, dead border, 100 generations.
2. **Fair comparison (20 seconds).** Same `N`, seed `42`, same initial grid, same rules. Two buffers so a generation is consistent.
3. **CPU (40 seconds).** Nested loops over the 2D grid, neighbour sum, write `nextGrid`, swap, time only that loop with `std::chrono`.
4. **CUDA (70 seconds).** One thread per cell, `16 x 16` blocks, index from `blockIdx` and `threadIdx`, kernel reads `d_current` and writes `d_next`, swap pointers instead of copying, grids stay on the GPU, CUDA events around the loop only.
5. **Proof (20 seconds).** Same living cells and same checksum. If they match, the GPU changed the speed and not the result.
6. **Speedup (30 seconds).** CPU time divided by CUDA time. Say that small grids can fail to benefit because launch overhead is large relative to the work.
