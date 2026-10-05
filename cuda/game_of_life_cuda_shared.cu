// ==============================================================================
// High Performance Computing - Assignment 1
// Conway's Game of Life: CUDA Shared Memory Optimization (Tiling with Halo)
// ==============================================================================
//
// Technical Characteristics & Optimization Analysis:
// 1. Shared Memory Tiling:
//    Threads in a 16x16 block cooperatively load an 18x18 tile into on-chip
//    shared memory (__shared__ int s_grid[18][18]), comprising the 16x16 internal
//    cells plus a 1-cell halo around all four borders.
//
// 2. Halo Loading:
//    - Internal: Each thread loads its corresponding cell into s_grid[ty+1][tx+1].
//    - Halo Borders: Top, bottom, left, right edges and the 4 corner cells are
//      loaded by the boundary threads with zero-boundary checks.
//
// 3. Synchronization:
//    __syncthreads() ensures the entire 18x18 tile is resident in fast on-chip
//    SRAM before any thread computes neighbor counts.
//
// 4. Stencil Evaluation from Shared Memory:
//    Each thread calculates all 8 neighbor reads directly from s_grid without
//    any additional DRAM accesses, dramatically reducing global memory traffic.
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

constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int MAX_GRID_SIZE = 8192;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

constexpr int TILE_X = 16;
constexpr int TILE_Y = 16;
constexpr int SHARED_DIM_X = TILE_X + 2; // 18
constexpr int SHARED_DIM_Y = TILE_Y + 2; // 18

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

// Device helper: Conway's rules
__device__ inline int nextCellStateDevice(int alive, int neighbors) {
    if (alive == 1 && (neighbors == 2 || neighbors == 3)) return 1;
    if (alive == 0 && neighbors == 3) return 1;
    return 0;
}

// Safe global memory reader with zero boundary condition
__device__ inline int readGlobalSafe(const int* grid, int N, int row, int col) {
    if (row >= 0 && row < N && col >= 0 && col < N) {
        return grid[row * N + col];
    }
    return 0;
}

// Shared-memory tiled kernel
__global__ void gameOfLifeSharedKernel(const int* __restrict__ d_current,
                                       int* __restrict__ d_next,
                                       int N) {
    __shared__ int s_tile[SHARED_DIM_Y][SHARED_DIM_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    int s_x = tx + 1;
    int s_y = ty + 1;

    // 1. Cooperative load of internal tile cell
    s_tile[s_y][s_x] = readGlobalSafe(d_current, N, row, col);

    // 2. Load Top and Bottom halos
    if (ty == 0) {
        s_tile[0][s_x] = readGlobalSafe(d_current, N, row - 1, col);
    }
    if (ty == TILE_Y - 1) {
        s_tile[TILE_Y + 1][s_x] = readGlobalSafe(d_current, N, row + 1, col);
    }

    // 3. Load Left and Right halos
    if (tx == 0) {
        s_tile[s_y][0] = readGlobalSafe(d_current, N, row, col - 1);
    }
    if (tx == TILE_X - 1) {
        s_tile[s_y][TILE_X + 1] = readGlobalSafe(d_current, N, row, col + 1);
    }

    // 4. Load the 4 Corners
    if (tx == 0 && ty == 0) {
        s_tile[0][0] = readGlobalSafe(d_current, N, row - 1, col - 1);
    }
    if (tx == TILE_X - 1 && ty == 0) {
        s_tile[0][TILE_X + 1] = readGlobalSafe(d_current, N, row - 1, col + 1);
    }
    if (tx == 0 && ty == TILE_Y - 1) {
        s_tile[TILE_Y + 1][0] = readGlobalSafe(d_current, N, row + 1, col - 1);
    }
    if (tx == TILE_X - 1 && ty == TILE_Y - 1) {
        s_tile[TILE_Y + 1][TILE_X + 1] = readGlobalSafe(d_current, N, row + 1, col + 1);
    }

    // Synchronize to ensure entire 18x18 tile is loaded into shared memory
    __syncthreads();

    // Out-of-bounds boundary guard for writing output
    if (row >= N || col >= N) {
        return;
    }

    // 5. Stencil evaluation directly from on-chip SRAM (s_tile)
    int neighbors = s_tile[s_y - 1][s_x - 1] + s_tile[s_y - 1][s_x] + s_tile[s_y - 1][s_x + 1] +
                    s_tile[s_y][s_x - 1]                            + s_tile[s_y][s_x + 1]     +
                    s_tile[s_y + 1][s_x - 1] + s_tile[s_y + 1][s_x] + s_tile[s_y + 1][s_x + 1];

    int alive = s_tile[s_y][s_x];
    d_next[row * N + col] = nextCellStateDevice(alive, neighbors);
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

    dim3 blockSize(TILE_X, TILE_Y);
    dim3 gridSize((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y);

    cudaEvent_t startKernel, stopKernel;
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));

    CUDA_CHECK(cudaMemcpy(d_current, hostGrid.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(startKernel));

    for (int it = 0; it < iterations; ++it) {
        gameOfLifeSharedKernel<<<gridSize, blockSize>>>(d_current, d_next, N);
        CUDA_CHECK(cudaGetLastError());

        int* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    CUDA_CHECK(cudaEventRecord(stopKernel));
    CUDA_CHECK(cudaEventSynchronize(stopKernel));

    CUDA_CHECK(cudaMemcpy(hostGrid.data(), d_current, bytes, cudaMemcpyDeviceToHost));

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, startKernel, stopKernel));

    const long long livingCells = countLivingCells(hostGrid);
    const unsigned long long checksum = computeChecksum(hostGrid);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------\n";
    std::cout << "Implementation: CUDA (Shared Memory Tiled)\n";
    std::cout << "GPU name: " << prop.name << "\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Tile size: " << TILE_X << " x " << TILE_Y << " (Shared tile: 18 x 18)\n";
    std::cout << "Kernel execution time (ms): " << kernelMs << "\n";
    std::cout << "Living cells: " << livingCells << "\n";
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "----------------------------------------\n";

    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));

    return 0;
}
