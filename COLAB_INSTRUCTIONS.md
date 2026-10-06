# Google Colab execution guide

Open the notebook in `notebooks/`, or paste these cells into a new Colab notebook in order. Choose a GPU runtime first.

# Conway's Game of Life: CPU versus CUDA
Select Runtime → Change runtime type → GPU, then run cells in order.
Both executables run in this same Linux GPU runtime; the Mac is only the editor.
The runner validates exact grids before the sweep, runs 100 generations three times
at each of five sizes, and exports means, samples, standard deviations and metadata.
Existing repository results are not evidence for this new run.

## Cell 1

```python
!nvidia-smi
!nvcc --version
!g++ --version
!lscpu

```

## Cell 2

```python
from pathlib import Path
for name in ["cpu", "cuda", "tests", "results"]:
    Path(name).mkdir(exist_ok=True)

```

## Cell 3

```python
%%writefile cpu/game_of_life_cpu.cpp
// ==============================================================================
// High Performance Computing - Assignment 1
// Conway's Game of Life: Sequential CPU Implementation
// ==============================================================================
//
// Technical Characteristics & Design Decisions:
// 1. Contiguous 1D Allocation for 2D Grid:
//    Instead of fragmented dynamic allocations (such as int** or vector<vector<int>>),
//    the N x N grid is allocated as a single contiguous memory block.
//    Cell (row, col) is mapped via the row-major formula: index = row * N + col.
//    Rationale:
//    - Exploits spatial and temporal cache locality (L1/L2 cache line prefetching).
//    - Eliminates multiple levels of pointer indirection (pointer chasing).
//    - Guarantees algorithmic and memory layout parity with the CUDA GPU version.
//
// 2. Zero Boundary Condition (Dead Borders):
//    Cells outside the N x N grid domain are strictly treated as dead (state = 0).
//
// 3. Double-Buffering / Pointer Swapping:
//    Two grids ('currentGrid' and 'nextGrid') are maintained. Each generation reads
//    from 'currentGrid' and writes to 'nextGrid', followed by an O(1) buffer pointer swap.
//    This prevents race conditions and avoids copying memory between generations.
//
// 4. Isolated Benchmark Timing:
//    Only the 100 simulation iterations are timed using std::chrono::steady_clock.
//    Memory allocation, grid initialization, validation, and I/O are strictly excluded.
// ==============================================================================

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <iostream>
#include <new>
#include <utility>
#include <vector>

// Benchmark Configuration & Constants
constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int MAX_GRID_SIZE = 8192;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " [grid_size] [iterations] [output_grid.bin]\n"
              << "Defaults: grid_size = " << DEFAULT_GRID_SIZE
              << ", iterations = " << DEFAULT_ITERATIONS << "\n"
              << "Example: " << program << " 1024 100\n";
}

// Parses positive command-line arguments safely
bool parsePositiveInt(const char* text, int& value, int maxValue) {
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);

    if (text[0] == '\0' || end == text || *end != '\0' || errno == ERANGE) {
        return false;
    }
    if (parsed <= 0 || parsed > maxValue) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

// Fast deterministic LCG PRNG (Numerical Recipes: a=1664525, c=1013904223).
// Using the most significant bit (state >> 31) ensures uniform 0 or 1 states.
inline int nextRandomCell(unsigned int& state) {
    state = 1664525u * state + 1013904223u;
    return static_cast<int>(state >> 31);
}

// Initializes the grid deterministically with seed 42 in row-major order.
void initializeGrid(int* grid, int N, unsigned int seed = RANDOM_SEED) {
    unsigned int state = seed;
    const size_t totalCells = static_cast<size_t>(N) * N;
    for (size_t i = 0; i < totalCells; ++i) {
        grid[i] = nextRandomCell(state);
    }
}

// Counts the 8 Moore neighbours for cell (row, col).
// Out-of-bound coordinates are treated as dead (zero boundary).
inline int countNeighbors(const int* grid, int N, int row, int col) {
    int liveNeighbors = 0;

    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;

            int nr = row + dr;
            int nc = col + dc;

            // Zero boundary check
            if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                liveNeighbors += grid[nr * N + nc];
            }
        }
    }

    return liveNeighbors;
}

// Applies Conway's four rules of life:
// 1. Live cell with < 2 live neighbours dies (underpopulation).
// 2. Live cell with 2 or 3 live neighbours survives.
// 3. Live cell with > 3 live neighbours dies (overpopulation).
// 4. Dead cell with exactly 3 live neighbours becomes alive (reproduction).
inline int nextCellState(int alive, int neighbors) {
    if (alive == 1 && (neighbors == 2 || neighbors == 3)) {
        return 1;
    }
    if (alive == 0 && neighbors == 3) {
        return 1;
    }
    return 0;
}

// Computes one generation from currentGrid into nextGrid.
void updateGrid(const int* currentGrid, int* nextGrid, int N) {
    for (int row = 0; row < N; ++row) {
        const int rowOffset = row * N;
        for (int col = 0; col < N; ++col) {
            int neighbors = countNeighbors(currentGrid, N, row, col);
            int alive = currentGrid[rowOffset + col];
            nextGrid[rowOffset + col] = nextCellState(alive, neighbors);
        }
    }
}

// Executes the simulation loop for the specified iterations using double-buffering.
void runSimulation(int*& currentGrid, int*& nextGrid, int N, int iterations) {
    for (int it = 0; it < iterations; ++it) {
        updateGrid(currentGrid, nextGrid, N);
        // O(1) buffer swap
        std::swap(currentGrid, nextGrid);
    }
}

// Counts total living cells in the grid for correctness validation.
long long countLivingCells(const int* grid, size_t cellCount) {
    long long living = 0;
    for (size_t i = 0; i < cellCount; ++i) {
        living += grid[i];
    }
    return living;
}

// 64-bit diagnostic checksum; exact validation requires comparing every cell.
unsigned long long computeChecksum(const int* grid, size_t cellCount) {
    unsigned long long checksum = 0;
    for (size_t i = 0; i < cellCount; ++i) {
        checksum = checksum * CHECKSUM_MULTIPLIER + static_cast<unsigned long long>(grid[i]);
    }
    return checksum;
}

#ifndef LIFE_NO_MAIN
int main(int argc, char* argv[]) {
    int N = DEFAULT_GRID_SIZE;
    int iterations = DEFAULT_ITERATIONS;

    if (argc > 4) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 2 && !parsePositiveInt(argv[1], N, MAX_GRID_SIZE)) {
        std::cerr << "Error: Grid size must be an integer between 1 and " << MAX_GRID_SIZE << ".\n";
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 3 && !parsePositiveInt(argv[2], iterations, INT_MAX)) {
        std::cerr << "Error: Iterations must be a positive integer.\n";
        printUsage(argv[0]);
        return 1;
    }

    const size_t cellCount = static_cast<size_t>(N) * static_cast<size_t>(N);

    // Contiguous memory allocation using std::vector to guarantee exception-safe cleanup
    std::vector<int> gridA;
    std::vector<int> gridB;

    try {
        gridA.resize(cellCount);
        gridB.resize(cellCount);
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: Insufficient host memory to allocate " << N << " x " << N << " grid.\n";
        return 1;
    }

    int* currentGrid = gridA.data();
    int* nextGrid = gridB.data();

    // Deterministic initialization (excluded from benchmark timing)
    initializeGrid(currentGrid, N, RANDOM_SEED);

    // ==========================================
    // Benchmark Timed Region: Simulation only
    // ==========================================
    const auto startTime = std::chrono::steady_clock::now();

    runSimulation(currentGrid, nextGrid, N, iterations);

    const auto endTime = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsedTime = endTime - startTime;
    // ==========================================

    // Optional exact-grid export, outside all timing intervals (one byte per cell).
    if (argc == 4) {
        std::ofstream output(argv[3], std::ios::binary);
        for (size_t i = 0; i < cellCount; ++i) output.put(static_cast<char>(currentGrid[i]));
        output.close();
        if (!output) { std::cerr << "Grid export failed\n"; return 1; }
    }

    // Validation metrics
    const long long livingCells = countLivingCells(currentGrid, cellCount);
    const unsigned long long checksum = computeChecksum(currentGrid, cellCount);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "----------------------------------------\n";
    std::cout << "Implementation: CPU\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Execution time (ms): " << elapsedTime.count() << "\n";
    std::cout << "Living cells: " << livingCells << "\n";
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "----------------------------------------\n";

    return 0;
}

#endif

```

## Cell 4

```python
%%writefile tests/test_cpu.cpp
#define LIFE_NO_MAIN
#include "../cpu/game_of_life_cpu.cpp"
#include <cassert>

// Independent reference: scatter each live cell's contribution to neighbours.
std::vector<int> reference(const std::vector<int>& a, int n) {
    std::vector<int> counts(n*n), out(n*n);
    for (int r=0;r<n;++r) for (int c=0;c<n;++c) if(a[r*n+c])
        for(int y=r-1;y<=r+1;++y) for(int x=c-1;x<=c+1;++x)
            if(y>=0 && y<n && x>=0 && x<n && (y!=r || x!=c)) ++counts[y*n+x];
    for(int i=0;i<n*n;++i) out[i]=counts[i]==3 || (a[i] && counts[i]==2);
    return out;
}
int main() {
    for(int alive=0;alive<=1;++alive) for(int neighbours=0;neighbours<=8;++neighbours)
        assert(nextCellState(alive,neighbours)==(neighbours==3 || (alive && neighbours==2)));
    // Stable block on the corner verifies edges remain active (only exterior is dead).
    std::vector<int> block={1,1,0,1,1,0,0,0,0}, next(9);
    updateGrid(block.data(),next.data(),3); assert(next==block);
    std::vector<int> blink(25), expected(25), output(25);
    blink[11]=blink[12]=blink[13]=1;
    expected[7]=expected[12]=expected[17]=1;
    updateGrid(blink.data(),output.data(),5); assert(output==expected);
    updateGrid(output.data(),expected.data(),5); assert(expected==blink);
    for(int n: {1,2,3,15,16,17,31,32,33,65}) {
        std::vector<int> a(n*n), b(n*n);
        initializeGrid(a.data(),n);
        auto ref=a;
        int* current=a.data(); int* scratch=b.data();
        for(int generation=0;generation<100;++generation) {
            ref=reference(ref,n);
            runSimulation(current,scratch,n,1);
            assert(std::equal(ref.begin(),ref.end(),current));
        }
    }
    std::cout << "CPU tests: PASSED (rules, patterns, boundaries, 100-generation reference)\n";
}

```

## Cell 5

```python
%%writefile cuda/game_of_life_cuda.cu
// ==============================================================================
// High Performance Computing - Assignment 1
// Conway's Game of Life: CUDA GPU Implementation (Global Memory Baseline)
// ==============================================================================
//
// Technical Characteristics & HPC Design Decisions:
// 1. Thread Mapping (1 Thread per Cell):
//    Each thread computes the next state of exactly one cell at (row, col):
//      int col = blockIdx.x * blockDim.x + threadIdx.x;
//      int row = blockIdx.y * blockDim.y + threadIdx.y;
//
// 2. 2D Block Geometry (16 x 16 = 256 threads):
//    - 256 threads per block is an exact multiple of warp size (32), giving 8 warps.
//    - Balances register pressure and active warp occupancy on Streaming Multiprocessors (SMs).
//    - Threads in the same warp advance along columns (threadIdx.x), reading consecutive
//      memory addresses in row-major layout (row * N + col).
//    - Each warp spans two 16-cell row segments; neighbour loads may be unaligned.
//      Actual transaction efficiency and occupancy require profiling.
//
// 3. Device Memory Persistence & Pointer Swapping:
//    Both current and next grids remain in GPU VRAM across all 100 iterations.
//    Pointers d_current and d_next are swapped on the host in O(1) time between launches.
//    Eliminates redundant Host-to-Device / Device-to-Host transfers every generation.
//
// 4. Dual Timing Instrumentation (Part 6 Requirement):
//    - Kernel / Simulation Time: Measured with cudaEvent_t strictly around the 100 iterations.
//    - Total End-to-End Time: Includes H2D transfer + 100 kernel launches + D2H transfer.
// ==============================================================================

#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <iostream>
#include <new>
#include <vector>

// Benchmark Configuration & Constants
constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int MAX_GRID_SIZE = 8192;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

// Block Dimensions: 16 x 16 = 256 threads per block
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

// Robust CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__        \
                      << " in " << #call << " -> "                              \
                      << cudaGetErrorString(error) << "\n";                     \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " [grid_size] [iterations] [output_grid.bin]\n"
              << "Defaults: grid_size = " << DEFAULT_GRID_SIZE
              << ", iterations = " << DEFAULT_ITERATIONS << "\n"
              << "Example: " << program << " 1024 100\n";
}

bool parsePositiveInt(const char* text, int& value, int maxValue) {
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);

    if (text[0] == '\0' || end == text || *end != '\0' || errno == ERANGE) {
        return false;
    }
    if (parsed <= 0 || parsed > maxValue) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

// PRNG identical to CPU version (Seed 42)
inline int nextRandomCell(unsigned int& state) {
    state = 1664525u * state + 1013904223u;
    return static_cast<int>(state >> 31);
}

void initializeGrid(std::vector<int>& grid, int N, unsigned int seed = RANDOM_SEED) {
    unsigned int state = seed;
    const size_t totalCells = static_cast<size_t>(N) * N;
    for (size_t i = 0; i < totalCells; ++i) {
        grid[i] = nextRandomCell(state);
    }
}

long long countLivingCells(const std::vector<int>& grid) {
    long long living = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        living += grid[i];
    }
    return living;
}

unsigned long long computeChecksum(const std::vector<int>& grid) {
    unsigned long long checksum = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        checksum = checksum * CHECKSUM_MULTIPLIER + static_cast<unsigned long long>(grid[i]);
    }
    return checksum;
}

// Device function: counts live neighbors for cell (row, col) with dead boundary condition
__device__ inline int countNeighborsDevice(const int* __restrict__ grid, int N, int row, int col) {
    int liveNeighbors = 0;

    #pragma unroll
    for (int dr = -1; dr <= 1; ++dr) {
        #pragma unroll
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;

            int nr = row + dr;
            int nc = col + dc;

            if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                liveNeighbors += grid[nr * N + nc];
            }
        }
    }

    return liveNeighbors;
}

// Device function: Conway's rules
__device__ inline int nextCellStateDevice(int alive, int neighbors) {
    if (alive == 1 && (neighbors == 2 || neighbors == 3)) {
        return 1;
    }
    if (alive == 0 && neighbors == 3) {
        return 1;
    }
    return 0;
}

// CUDA Kernel: 1 thread per cell
// Consecutive threadIdx.x access adjacent columns in row-major layout -> coalesced memory load
__global__ void gameOfLifeKernel(const int* __restrict__ d_current,
                                int* __restrict__ d_next,
                                int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for grids not evenly divisible by block size
    if (row >= N || col >= N) {
        return;
    }

    int idx = row * N + col;
    int neighbors = countNeighborsDevice(d_current, N, row, col);
    int alive = d_current[idx];

    d_next[idx] = nextCellStateDevice(alive, neighbors);
}

int main(int argc, char* argv[]) {
    int N = DEFAULT_GRID_SIZE;
    int iterations = DEFAULT_ITERATIONS;

    if (argc > 4) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 2 && !parsePositiveInt(argv[1], N, MAX_GRID_SIZE)) {
        std::cerr << "Error: Grid size must be between 1 and " << MAX_GRID_SIZE << ".\n";
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 3 && !parsePositiveInt(argv[2], iterations, INT_MAX)) {
        std::cerr << "Error: Iterations must be a positive integer.\n";
        printUsage(argv[0]);
        return 1;
    }

    const size_t cellCount = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = cellCount * sizeof(int);

    // Host memory allocation
    std::vector<int> hostGrid(cellCount);
    initializeGrid(hostGrid, N, RANDOM_SEED);

    // Query GPU device properties
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable GPU detected.\n";
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Allocate GPU device memory
    int* d_current = nullptr;
    int* d_next = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_current), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next), bytes));

    // Configure 2D grid of thread blocks
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((N + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                  (N + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Create CUDA timing events
    cudaEvent_t startKernel, stopKernel;
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));

    // Untimed warm-up in this process; restore initial state before measurement.
    CUDA_CHECK(cudaMemcpy(d_current, hostGrid.data(), bytes, cudaMemcpyHostToDevice));
    gameOfLifeKernel<<<gridSize, blockSize>>>(d_current, d_next, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // Measure Total End-to-End Time: H2D + Kernels + D2H
    // -------------------------------------------------------------
    const auto wallStart = std::chrono::steady_clock::now();

    // Initial Host-to-Device transfer
    CUDA_CHECK(cudaMemcpy(d_current, hostGrid.data(), bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------
    // Measure Kernel Simulation Time: 100 iterations loop only
    // -------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(startKernel));

    for (int it = 0; it < iterations; ++it) {
        gameOfLifeKernel<<<gridSize, blockSize>>>(d_current, d_next, N);
        CUDA_CHECK(cudaGetLastError());

        // Pointer swap on host (O(1)) - data stays resident in GPU VRAM
        int* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    CUDA_CHECK(cudaEventRecord(stopKernel));
    CUDA_CHECK(cudaEventSynchronize(stopKernel));

    // Final Device-to-Host transfer (d_current contains the latest generation)
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), d_current, bytes, cudaMemcpyDeviceToHost));

    const auto wallStop = std::chrono::steady_clock::now();

    // Compute elapsed times
    float kernelMs = 0.0f;
    const double totalMs = std::chrono::duration<double, std::milli>(wallStop-wallStart).count();
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, startKernel, stopKernel));


    // Verification metrics
    // Optional exact-grid export, outside all timing intervals (one byte per cell).
    if (argc == 4) {
        std::ofstream output(argv[3], std::ios::binary);
        for (size_t i = 0; i < cellCount; ++i) output.put(static_cast<char>(hostGrid.data()[i]));
        output.close();
        if (!output) { std::cerr << "Grid export failed\n"; return 1; }
    }

    const long long livingCells = countLivingCells(hostGrid);
    const unsigned long long checksum = computeChecksum(hostGrid);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "----------------------------------------\n";
    std::cout << "Implementation: CUDA (Global Memory)\n";
    std::cout << "GPU name: " << prop.name << "\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Block size: " << BLOCK_DIM_X << " x " << BLOCK_DIM_Y << "\n";
    std::cout << "Kernel execution time (ms): " << kernelMs << "\n";
    std::cout << "Total GPU time (inc. transfers) (ms): " << totalMs << "\n";
    std::cout << "Living cells: " << livingCells << "\n";
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "----------------------------------------\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));

    return 0;
}

```

## Cell 6

```python
%%writefile run_benchmarks.py
#!/usr/bin/env python3
"""Run on the Colab GPU runtime: exact validation, then 100-generation benchmarks."""
import csv
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIZES = [256, 512, 1024, 2048, 4096]
ITERATIONS = 100
RUNS = 3


def command(args):
    return subprocess.run([str(x) for x in args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def fields(output):
    return dict(line.split(': ', 1) for line in output.splitlines() if ': ' in line)


def timing(values, key):
    value = float(values[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid measurement: {key}={value}')
    return value


def validate(cpu_path, gpu_path, cells):
    a, b = cpu_path.read_bytes(), gpu_path.read_bytes()
    if len(a) != cells or len(b) != cells:
        raise RuntimeError('Incomplete grid output')
    mismatches = sum(x != y for x, y in zip(a, b))
    if mismatches:
        raise RuntimeError(f'Validation: FAILED ({mismatches} mismatched cells)')
    print(f'Validation: PASSED ({cells} cells, 0 mismatches)', flush=True)


def main():
    if platform.system() != 'Linux' or not shutil.which('nvcc'):
        raise SystemExit('Run this comparison in a Linux NVIDIA GPU runtime (Google Colab). Existing results were not changed.')
    gpu_info = command(['nvidia-smi'])
    environment = {
        'timestampUTC': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'cpu': command(['lscpu']),
        'gpu': gpu_info, 'cudaCompiler': command(['nvcc', '--version']),
        'cpuCompiler': command(['g++', '--version']),
        'cpuFlags': '-O3 -std=c++17', 'cudaFlags': '-O3 -std=c++17',
        'gridSizes': SIZES, 'iterations': ITERATIONS, 'runs': RUNS,
        'seed': 42, 'boundary': 'outside-domain cells dead',
        'cpuThreads': 1, 'blockSize': [16, 16],
        'validation': 'exact byte-per-cell comparison before timing sweep and every repetition',
        'gpuTotalTiming': 'steady_clock: H2D + simulation + synchronized D2H; excludes allocation/context/warm-up',
        'gpuSimulationTiming': 'CUDA events around 100 launches; includes device timeline gaps between launches',
        'sourceSHA256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT/'cpu/game_of_life_cpu.cpp', ROOT/'cuda/game_of_life_cuda.cu']}
    }
    # Temporary binaries avoid accidentally executing stale binaries after a compile failure.
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        cpu, gpu, tests = [temp / name for name in ('cpu', 'gpu', 'tests')]
        command(['g++', '-O3', '-std=c++17', 'tests/test_cpu.cpp', '-o', tests])
        print(command([tests]), flush=True)
        command(['g++', '-O3', '-std=c++17', 'cpu/game_of_life_cpu.cpp', '-o', cpu])
        command(['nvcc', '-O3', '-std=c++17', 'cuda/game_of_life_cuda.cu', '-o', gpu])
        a, b = temp/'cpu.bin', temp/'gpu.bin'
        # Odd/even iterations exercise pointer parity; nonmultiples exercise partial blocks.
        for size in [1, 2, 3, 15, 16, 17, 31, 33, 65]:
            for generations in [1, 2, 100]:
                command([cpu, size, generations, a])
                command([gpu, size, generations, b])
                validate(a, b, size*size)
        # Verify all assignment sizes before starting the measured sweep.
        for size in SIZES:
            command([cpu, size, ITERATIONS, a])
            command([gpu, size, ITERATIONS, b])
            validate(a, b, size*size)
        records = []
        for size in SIZES:
            samples = []
            for repetition in range(RUNS):
                outputs = {}
                # Alternate execution order to reduce systematic ordering bias.
                order = [('cpu', cpu, a), ('gpu', gpu, b)]
                if repetition % 2: order.reverse()
                for name, executable, destination in order:
                    outputs[name] = fields(command([executable, size, ITERATIONS, destination]))
                validate(a, b, size*size)
                c, g = outputs['cpu'], outputs['gpu']
                samples.append({'cpuTimeMs': timing(c, 'Execution time (ms)'),
                                'gpuKernelTimeMs': timing(g, 'Kernel execution time (ms)'),
                                'gpuTotalTimeMs': timing(g, 'Total GPU time (inc. transfers) (ms)')})
            means = {key: statistics.mean(row[key] for row in samples) for key in samples[0]}
            record = {'gridSize': size, 'cells': size*size, 'iterations': ITERATIONS,
                      'runs': RUNS, 'blockSize': '16x16', **means,
                      'speedup': means['cpuTimeMs']/means['gpuKernelTimeMs'],
                      'speedupTotal': means['cpuTimeMs']/means['gpuTotalTimeMs'],
                      'validation': 'PASSED', 'mismatchedCells': 0,
                      'gpuName': g['GPU name'], 'checksum': c['Checksum'],
                      'livingCells': int(c['Living cells']), 'samples': samples,
                      'stddevMs': {key: statistics.stdev(row[key] for row in samples) for key in means}}
            records.append(record)
            print(f'{size}x{size}: CPU {means["cpuTimeMs"]:.3f} ms, CUDA {means["gpuKernelTimeMs"]:.3f} ms, speedup {record["speedup"]:.2f}x', flush=True)
    # Export only after the entire sweep succeeds. No partial or CPU-only comparison.
    destination = ROOT/'results'
    destination.mkdir(exist_ok=True)
    (destination/'benchmark_results.json').write_text(json.dumps(records, indent=2)+'\n')
    (destination/'benchmark_environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    columns = [key for key in records[0] if key not in ('samples', 'stddevMs')]
    with (destination/'benchmark_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print('Exported validated results and environment metadata to results/.')


if __name__ == '__main__':
    main()

```

## Cell 7

```python
import subprocess
subprocess.run(["python3", "run_benchmarks.py"], check=True)

```

## Cell 8

```python
import json
from pathlib import Path
from google.colab import files
rows = json.loads(Path("results/benchmark_results.json").read_text())
assert len(rows) == 5 and all(r["validation"] == "PASSED" for r in rows)
for name in ["benchmark_results.json", "benchmark_results.csv", "benchmark_environment.json"]:
    files.download("results/" + name)

```

## Timing and interpretation

CPU simulation uses a monotonic host clock. CUDA simulation uses events around
100 kernel launches. CUDA total uses a host clock around the initial copy,
simulation, synchronization and final copy. Allocation, initialization, context
creation, warm-up, validation and file output are excluded. Each CUDA process
warms its kernel before restoring the initial grid. CPU measurements include
ordinary first-iteration cache effects; neither measurement includes compilation.

The 16×16 block contains 256 threads (eight warps). A warp spans two row segments;
this supports adjacent-column accesses but does not guarantee perfect memory
transactions or optimal occupancy. Shared memory is an optional experiment in
`cuda/game_of_life_cuda_shared.cu`, not part of this validated baseline sweep.

Same-session execution improves comparability but Colab load and clock variation
remain uncontrolled. Record the exported CPU, GPU, compiler and CUDA details.
Speedup is the ratio of mean CPU time to mean GPU time, separately for simulation
and total time. Do not infer bandwidth saturation or monotonic scaling without
evidence. The current runner stops on compilation, execution or validation failure.

After the run, return all three exported files for analysis, dashboard integration,
and final report/video revisions. CUDA has not been verified on the Mac.

References: [Game of Life rules](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life),
[NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).
