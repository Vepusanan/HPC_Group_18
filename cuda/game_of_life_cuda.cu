// Conway's Game of Life on an NVIDIA GPU.
//
// One thread computes one cell. The two grids stay in device memory for all
// generations. This program must use the same seed, initialisation, rules,
// borders, and iteration count as game_of_life_cpu.cpp. Matching living-cell
// counts and checksums show that the GPU did not change the result.

#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <vector>

// Shared with the CPU program. Do not change one without changing the other.
constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int MAX_GRID_SIZE = 8192;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

// Assignment launch shape: a 16 x 16 thread block. 256 threads per block.
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

// Checks a CUDA call and stops with the location and message if it failed.
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " in " << #call << ": "                                \
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

int nextRandomCell(unsigned int& state) {
    state = 1664525u * state + 1013904223u;
    return static_cast<int>(state >> 31);
}

// Host-side initialisation. It is intentionally the same nested loop as the CPU
// program. The GPU does not build the random grid; it receives a copy of it.
void initializeGrid(std::vector<int>& grid, int N) {
    unsigned int state = RANDOM_SEED;

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            grid[row * N + col] = nextRandomCell(state);
        }
    }
}

long long countLivingCells(const std::vector<int>& grid) {
    long long livingCells = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        livingCells += grid[i];
    }
    return livingCells;
}

unsigned long long computeChecksum(const std::vector<int>& grid) {
    unsigned long long checksum = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        checksum = checksum * CHECKSUM_MULTIPLIER +
                   static_cast<unsigned long long>(grid[i]);
    }
    return checksum;
}

// Device copy of the CPU neighbour count. Runs on the GPU, for one cell.
// Cells outside the grid are dead under the zero boundary condition.
__device__ int countNeighbours(const int* grid, int N, int row, int col) {
    int neighbours = 0;

    for (int deltaRow = -1; deltaRow <= 1; ++deltaRow) {
        for (int deltaCol = -1; deltaCol <= 1; ++deltaCol) {
            if (deltaRow == 0 && deltaCol == 0) {
                continue;
            }

            int neighbourRow = row + deltaRow;
            int neighbourCol = col + deltaCol;

            if (neighbourRow < 0 || neighbourRow >= N ||
                neighbourCol < 0 || neighbourCol >= N) {
                continue;
            }

            neighbours += grid[neighbourRow * N + neighbourCol];
        }
    }

    return neighbours;
}

// Same rules as nextCellState() in the CPU program.
__device__ int nextCellState(int alive, int neighbours) {
    if (alive == 1 && (neighbours == 2 || neighbours == 3)) {
        return 1;
    }
    if (alive == 0 && neighbours == 3) {
        return 1;
    }
    return 0;
}

// One thread, one cell.
//
// There is no __syncthreads() here. Each thread reads the current grid and
// writes only its own cell of the next grid, so threads do not conflict.
// The following generation is another launch, and that launch waits for this
// one because both are placed on the same CUDA stream.
__global__ void gameOfLifeKernel(const int* d_current, int* d_next, int N) {
    // CUDA thread indexing for a 2D block.
    // x selects the column. y selects the row.
    int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    // The block grid is rounded up, so edge threads may fall outside an N
    // that is not a multiple of 16. Those threads have no cell to update.
    if (row >= N || col >= N) {
        return;
    }

    // Consecutive threadIdx.x values own consecutive columns. Columns are
    // adjacent in this row-major layout, so those global loads are coalesced.
    int neighbours = countNeighbours(d_current, N, row, col);
    int alive = d_current[row * N + col];
    d_next[row * N + col] = nextCellState(alive, neighbours);
}

int main(int argc, char* argv[]) {
    int N = DEFAULT_GRID_SIZE;
    int iterations = DEFAULT_ITERATIONS;

    if (argc > 3) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 2 && !parsePositiveInt(argv[1], N, MAX_GRID_SIZE)) {
        std::cerr << "Grid size must be an integer from 1 to " << MAX_GRID_SIZE << ".\n";
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 3 && !parsePositiveInt(argv[2], iterations, INT_MAX)) {
        std::cerr << "Iterations must be a positive integer.\n";
        printUsage(argv[0]);
        return 1;
    }

    const size_t cellCount = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = cellCount * sizeof(int);

    std::vector<int> hostGrid;
    try {
        hostGrid.resize(cellCount);
    } catch (const std::bad_alloc&) {
        std::cerr << "Not enough memory for a " << N << " x " << N << " grid.\n";
        return 1;
    }

    // Same initial grid as the CPU program. This is outside the GPU timer.
    initializeGrid(hostGrid, N);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU was found.\n";
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    // Device memory: two full grids. They stay on the GPU for every generation.
    int* d_current = nullptr;
    int* d_next = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_current), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next), bytes));

    // Initial copy is setup, so it is done before the timed section.
    CUDA_CHECK(cudaMemcpy(d_current, hostGrid.data(), bytes, cudaMemcpyHostToDevice));

    // CUDA grid/block configuration.
    // block is 16 x 16 threads. grid is the number of blocks needed to cover N.
    // The ceiling division rounds up when N is not a multiple of the block size.
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X,
              (N + BLOCK_Y - 1) / BLOCK_Y);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Timed section: the generation loop only.
    // The start event is recorded after the host-to-device copy has been queued
    // on this same stream, so the clock starts once that copy has finished.
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int generation = 0; generation < iterations; ++generation) {
        // Kernel execution: one launch updates every cell into the other buffer.
        gameOfLifeKernel<<<grid, block>>>(d_current, d_next, N);
        CUDA_CHECK(cudaGetLastError());

        // Pointer swapping. The cell data is not copied. The next launch reads
        // the buffer just written and writes into the previous one.
        // Kernel arguments are captured at launch, so this swap does not change
        // the kernel that was just submitted.
        int* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaGetLastError());

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    // Final copy is outside the timer. d_current is the latest generation
    // because the loop swaps the two pointers once per iteration.
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), d_current, bytes, cudaMemcpyDeviceToHost));

    const long long livingCells = countLivingCells(hostGrid);
    const unsigned long long checksum = computeChecksum(hostGrid);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------\n";
    std::cout << "Implementation: CUDA\n";
    std::cout << "GPU name: " << deviceProperties.name << "\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Block size: " << BLOCK_X << " x " << BLOCK_Y << "\n";
    std::cout << "Execution time (ms): " << static_cast<double>(milliseconds) << "\n";
    std::cout << "Living cells: " << livingCells << "\n";
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "----------------------------------------\n";

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));

    return 0;
}
