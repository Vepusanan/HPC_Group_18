# Google Colab Execution Guide — Conway's Game of Life HPC Benchmark
**High Performance Computing — Group 18**

Because you are using an Apple Silicon Mac (M1), CUDA cannot run natively on macOS. We use **Google Colab** with an **NVIDIA GPU** for both the CPU and CUDA implementations. This ensures a **100% fair and consistent benchmark environment** where both the CPU baseline and GPU acceleration execute on the same physical server instance.

---

## Option A: Upload the Pre-Built Notebook (Fastest)

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select the file from this repository:
   `notebooks/Conway_Game_of_Life_HPC_Colab.ipynb`
3. Set the runtime to GPU:
   - In the top menu, go to **Runtime** → **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** (or any available NVIDIA GPU).
   - Click **Save**.
4. In the menu, click **Runtime** → **Run all** (or run cells one by one).
5. At the end of the run, `benchmark_results.json` and `benchmark_results.csv` will automatically download to your computer!

---

## Option B: Copy & Paste Cell-by-Cell into a New Colab Notebook

If you prefer to create a clean notebook from scratch, follow these exact steps:

### 1. Open Google Colab & Enable GPU
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **New Notebook**.
3. Go to **Runtime** → **Change runtime type** → select **T4 GPU** → click **Save**.

---

### Cell 1 (Code): Check Hardware Environment
```bash
# Verify GPU allocation and VRAM
!nvidia-smi

# Check CUDA compiler version
!nvcc --version

# Check CPU specifications in this Colab instance
!lscpu | grep -E "Model name|CPU\(s\):|Thread\(s\) per core:|CPU MHz"
```

---

### Cell 2 (Code): Write the CPU Implementation
```cpp
%%writefile game_of_life_cpu.cpp
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <utility>
#include <vector>

constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int MAX_GRID_SIZE = 8192;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

bool parsePositiveInt(const char* text, int& value, int maxValue) {
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0' || errno == ERANGE) return false;
    if (parsed <= 0 || parsed > maxValue) return false;
    value = static_cast<int>(parsed);
    return true;
}

inline int nextRandomCell(unsigned int& state) {
    state = 1664525u * state + 1013904223u;
    return static_cast<int>(state >> 31);
}

void initializeGrid(int* grid, int N, unsigned int seed = RANDOM_SEED) {
    unsigned int state = seed;
    const size_t totalCells = static_cast<size_t>(N) * N;
    for (size_t i = 0; i < totalCells; ++i) {
        grid[i] = nextRandomCell(state);
    }
}

inline int countNeighbors(const int* grid, int N, int row, int col) {
    int liveNeighbors = 0;
    for (int dr = -1; dr <= 1; ++dr) {
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

inline int nextCellState(int alive, int neighbors) {
    if (alive == 1 && (neighbors == 2 || neighbors == 3)) return 1;
    if (alive == 0 && neighbors == 3) return 1;
    return 0;
}

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

void runSimulation(int*& currentGrid, int*& nextGrid, int N, int iterations) {
    for (int it = 0; it < iterations; ++it) {
        updateGrid(currentGrid, nextGrid, N);
        std::swap(currentGrid, nextGrid);
    }
}

long long countLivingCells(const int* grid, size_t cellCount) {
    long long living = 0;
    for (size_t i = 0; i < cellCount; ++i) living += grid[i];
    return living;
}

unsigned long long computeChecksum(const int* grid, size_t cellCount) {
    unsigned long long checksum = 0;
    for (size_t i = 0; i < cellCount; ++i) {
        checksum = checksum * CHECKSUM_MULTIPLIER + static_cast<unsigned long long>(grid[i]);
    }
    return checksum;
}

int main(int argc, char* argv[]) {
    int N = DEFAULT_GRID_SIZE;
    int iterations = DEFAULT_ITERATIONS;
    if (argc >= 2 && !parsePositiveInt(argv[1], N, MAX_GRID_SIZE)) return 1;
    if (argc >= 3 && !parsePositiveInt(argv[2], iterations, INT_MAX)) return 1;

    const size_t cellCount = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<int> gridA(cellCount);
    std::vector<int> gridB(cellCount);
    int* currentGrid = gridA.data();
    int* nextGrid = gridB.data();

    initializeGrid(currentGrid, N, RANDOM_SEED);

    const auto startTime = std::chrono::high_resolution_clock::now();
    runSimulation(currentGrid, nextGrid, N, iterations);
    const auto endTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsedTime = endTime - startTime;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------\n";
    std::cout << "Implementation: CPU\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Execution time (ms): " << elapsedTime.count() << "\n";
    std::cout << "Living cells: " << countLivingCells(currentGrid, cellCount) << "\n";
    std::cout << "Checksum: " << computeChecksum(currentGrid, cellCount) << "\n";
    std::cout << "----------------------------------------\n";
    return 0;
}
```

---

### Cell 3 (Code): Write the CUDA Implementation
```cpp
%%writefile game_of_life_cuda.cu
#include <cuda_runtime.h>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int MAX_GRID_SIZE = 8192;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " in " << #call << " -> " \
                      << cudaGetErrorString(error) << "\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

bool parsePositiveInt(const char* text, int& value, int maxValue) {
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0' || errno == ERANGE) return false;
    if (parsed <= 0 || parsed > maxValue) return false;
    value = static_cast<int>(parsed);
    return true;
}

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
    for (size_t i = 0; i < grid.size(); ++i) living += grid[i];
    return living;
}

unsigned long long computeChecksum(const std::vector<int>& grid) {
    unsigned long long checksum = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        checksum = checksum * CHECKSUM_MULTIPLIER + static_cast<unsigned long long>(grid[i]);
    }
    return checksum;
}

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

__device__ inline int nextCellStateDevice(int alive, int neighbors) {
    if (alive == 1 && (neighbors == 2 || neighbors == 3)) return 1;
    if (alive == 0 && neighbors == 3) return 1;
    return 0;
}

__global__ void gameOfLifeKernel(const int* __restrict__ d_current,
                                int* __restrict__ d_next,
                                int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;

    int idx = row * N + col;
    int neighbors = countNeighborsDevice(d_current, N, row, col);
    int alive = d_current[idx];
    d_next[idx] = nextCellStateDevice(alive, neighbors);
}

int main(int argc, char* argv[]) {
    int N = DEFAULT_GRID_SIZE;
    int iterations = DEFAULT_ITERATIONS;
    if (argc >= 2 && !parsePositiveInt(argv[1], N, MAX_GRID_SIZE)) return 1;
    if (argc >= 3 && !parsePositiveInt(argv[2], iterations, INT_MAX)) return 1;

    const size_t cellCount = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = cellCount * sizeof(int);

    std::vector<int> hostGrid(cellCount);
    initializeGrid(hostGrid, N, RANDOM_SEED);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) return 1;

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int* d_current = nullptr;
    int* d_next = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_current), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next), bytes));

    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((N + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (N + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    cudaEvent_t startKernel, stopKernel, startTotal, stopTotal;
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&startTotal));
    CUDA_CHECK(cudaEventCreate(&stopTotal));

    CUDA_CHECK(cudaEventRecord(startTotal));
    CUDA_CHECK(cudaMemcpy(d_current, hostGrid.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startKernel));

    for (int it = 0; it < iterations; ++it) {
        gameOfLifeKernel<<<gridSize, blockSize>>>(d_current, d_next, N);
        CUDA_CHECK(cudaGetLastError());
        int* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    CUDA_CHECK(cudaEventRecord(stopKernel));
    CUDA_CHECK(cudaEventSynchronize(stopKernel));
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), d_current, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stopTotal));
    CUDA_CHECK(cudaEventSynchronize(stopTotal));

    float kernelMs = 0.0f, totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, startKernel, stopKernel));
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, startTotal, stopTotal));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------\n";
    std::cout << "Implementation: CUDA (Global Memory)\n";
    std::cout << "GPU name: " << prop.name << "\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Block size: " << BLOCK_DIM_X << " x " << BLOCK_DIM_Y << "\n";
    std::cout << "Kernel execution time (ms): " << kernelMs << "\n";
    std::cout << "Total GPU time (inc. transfers) (ms): " << totalMs << "\n";
    std::cout << "Living cells: " << countLivingCells(hostGrid) << "\n";
    std::cout << "Checksum: " << computeChecksum(hostGrid) << "\n";
    std::cout << "----------------------------------------\n";

    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaEventDestroy(startTotal));
    CUDA_CHECK(cudaEventDestroy(stopTotal));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    return 0;
}
```

---

### Cell 4 (Code): Compile with `-O3`
```bash
# Compile CPU binary
!g++ -O3 -std=c++17 game_of_life_cpu.cpp -o game_of_life_cpu

# Compile CUDA binary
!nvcc -O3 game_of_life_cuda.cu -o game_of_life_cuda
```

---

### Cell 5 (Code): Quick Correctness Validation
```bash
# Verify 256x256 for 100 iterations (both checksums and living cells must match exactly)
!./game_of_life_cpu 256 100
!./game_of_life_cuda 256 100
```

---

### Cell 6 (Code): Run Full Automated Benchmark Suite
```python
import subprocess, json, statistics, os

GRID_SIZES = [256, 512, 1024, 2048, 4096]
ITERATIONS = 100
RUNS = 3

def parse_out(out):
    gpu_name = "NVIDIA GPU"
    kernel_t, total_t, living, chk = None, None, None, None
    for line in out.splitlines():
        if "GPU name:" in line: gpu_name = line.split(":")[1].strip()
        elif "Kernel execution time (ms):" in line: kernel_t = float(line.split(":")[1].strip())
        elif "Total GPU time" in line: total_t = float(line.split(":")[1].strip())
        elif "Execution time (ms):" in line and kernel_t is None: kernel_t = float(line.split(":")[1].strip())
        elif "Living cells:" in line: living = int(line.split(":")[1].strip())
        elif "Checksum:" in line: chk = line.split(":")[1].strip()
    if total_t is None: total_t = kernel_t
    return gpu_name, kernel_t, total_t, living, chk

records = []
print(f"Executing benchmarks across {GRID_SIZES} ({RUNS} runs each)...\n")

for N in GRID_SIZES:
    cells = N * N
    print(f">>> Grid {N}x{N} ({cells:,} cells)")
    
    c_times = []
    c_living, c_chk = None, None
    for _ in range(RUNS):
        res = subprocess.run(f"./game_of_life_cpu {N} {ITERATIONS}", shell=True, capture_output=True, text=True)
        _, t, _, l, c = parse_out(res.stdout)
        c_times.append(t)
        c_living, c_chk = l, c
    avg_cpu = statistics.mean(c_times)
    
    k_times, tot_times = [], []
    g_living, g_chk, gpu_name = None, None, ""
    for _ in range(RUNS):
        res = subprocess.run(f"./game_of_life_cuda {N} {ITERATIONS}", shell=True, capture_output=True, text=True)
        g_name, kt, tt, l, c = parse_out(res.stdout)
        gpu_name = g_name
        k_times.append(kt)
        tot_times.append(tt)
        g_living, g_chk = l, c
    avg_kernel = statistics.mean(k_times)
    avg_total = statistics.mean(tot_times)
    
    matches = (c_living == g_living) and (c_chk == g_chk)
    status = "PASSED" if matches else "MISMATCH"
    sp_k = round(avg_cpu / avg_kernel, 2)
    sp_tot = round(avg_cpu / avg_total, 2)
    
    print(f"    CPU: {avg_cpu:.2f} ms | CUDA Kernel: {avg_kernel:.2f} ms | CUDA Total: {avg_total:.2f} ms")
    print(f"    Validation: {status} | Kernel Speedup: {sp_k}x | End-to-End Speedup: {sp_tot}x\n")
    
    records.append({
        "gridSize": N,
        "cells": cells,
        "iterations": ITERATIONS,
        "blockSize": "16x16",
        "cpuTimeMs": round(avg_cpu, 3),
        "gpuKernelTimeMs": round(avg_kernel, 3),
        "gpuTotalTimeMs": round(avg_total, 3),
        "speedup": sp_k,
        "speedupTotal": sp_tot,
        "livingCells": c_living,
        "checksum": str(c_chk),
        "validation": status,
        "gpuName": gpu_name
    })

with open("benchmark_results.json", "w") as f:
    json.dump(records, f, indent=2)

with open("benchmark_results.csv", "w") as f:
    f.write("grid_size,cells,iterations,block_size,cpu_time_ms,cuda_kernel_time_ms,cuda_total_time_ms,speedup_kernel,speedup_total,living_cells,checksum,validation,gpu_name\n")
    for r in records:
        f.write(f"{r['gridSize']},{r['cells']},{r['iterations']},{r['blockSize']},{r['cpuTimeMs']},{r['gpuKernelTimeMs']},{r['gpuTotalTimeMs']},{r['speedup']},{r['speedupTotal']},{r['livingCells']},{r['checksum']},{r['validation']},{r['gpuName']}\n")

print("✓ Benchmarks finished successfully!")
```

---

### Cell 7 (Code): Display Table & Download Files
```python
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

df = pd.read_csv("benchmark_results.csv")
print("=== OFFICIAL BENCHMARK RESULTS ===")
display(df[["grid_size", "cells", "cpu_time_ms", "cuda_kernel_time_ms", "cuda_total_time_ms", "speedup_kernel", "speedup_total", "validation"]])

# Download files to your computer if running inside Google Colab
try:
    import importlib
    colab_files = importlib.import_module("google.colab.files")
    colab_files.download("benchmark_results.json")
    colab_files.download("benchmark_results.csv")
except (ImportError, ModuleNotFoundError):
    print("Files saved in current directory: benchmark_results.json, benchmark_results.csv")
```

Once downloaded, place them in the `results/` folder of this project to update the interactive website and report!
