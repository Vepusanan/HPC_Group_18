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
//    - Enables 100% coalesced 128-byte global memory transactions.
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
    std::cerr << "Usage: " << program << " [grid_size] [iterations]\n"
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

    if (argc > 3) {
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
    cudaEvent_t startTotal, stopTotal;
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&startTotal));
    CUDA_CHECK(cudaEventCreate(&stopTotal));

    // -------------------------------------------------------------
    // Measure Total End-to-End Time: H2D + Kernels + D2H
    // -------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(startTotal));

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

    CUDA_CHECK(cudaEventRecord(stopTotal));
    CUDA_CHECK(cudaEventSynchronize(stopTotal));

    // Compute elapsed times
    float kernelMs = 0.0f;
    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, startKernel, stopKernel));
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, startTotal, stopTotal));

    // Verification metrics
    const long long livingCells = countLivingCells(hostGrid);
    const unsigned long long checksum = computeChecksum(hostGrid);

    std::cout << std::fixed << std::setprecision(3);
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
    CUDA_CHECK(cudaEventDestroy(startTotal));
    CUDA_CHECK(cudaEventDestroy(stopTotal));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));

    return 0;
}
