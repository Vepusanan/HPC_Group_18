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
