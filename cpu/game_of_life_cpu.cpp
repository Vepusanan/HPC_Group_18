// Sequential Conway's Game of Life on the CPU.
//
// This is the baseline for the CUDA version. Both programs use the same grid
// size, random seed, initialisation order, rules, dead borders, and iteration
// count. The living-cell count and checksum of the final grid should match.

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

// Shared with the CUDA program. Do not change one without changing the other.
constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int MAX_GRID_SIZE = 8192;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " [grid_size] [iterations]\n"
              << "Defaults: grid_size = " << DEFAULT_GRID_SIZE
              << ", iterations = " << DEFAULT_ITERATIONS << "\n"
              << "Example: " << program << " 1024 100\n";
}

// Accepts a positive integer up to maxValue. Rejects words, decimals, and signs.
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

// One step of the fixed generator. The top bit is used because the low bits of
// this kind of generator are less mixed. Alive is 1 and dead is 0.
int nextRandomCell(unsigned int& state) {
    state = 1664525u * state + 1013904223u;
    return static_cast<int>(state >> 31);
}

// Fills an N x N grid in row-major order.
// Cell (row, col) is stored at index row * N + col.
// The same loop and seed are used in the CUDA program, so both start identical.
void initializeGrid(std::vector<int>& grid, int N) {
    unsigned int state = RANDOM_SEED;

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            grid[row * N + col] = nextRandomCell(state);
        }
    }
}

// Counts the eight surrounding cells. Anything outside the grid is dead, so it
// is skipped. Living cells are stored as 1, so adding the stored value counts them.
int countNeighbours(const std::vector<int>& grid, int N, int row, int col) {
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

// Conway's rules for one cell:
//   - a live cell with 2 or 3 live neighbours survives
//   - a dead cell with exactly 3 live neighbours becomes alive
//   - every other cell is dead in the next generation
int nextCellState(int alive, int neighbours) {
    if (alive == 1 && (neighbours == 2 || neighbours == 3)) {
        return 1;
    }
    if (alive == 0 && neighbours == 3) {
        return 1;
    }
    return 0;
}

// Reads currentGrid and writes a full generation into nextGrid.
// The two grids are separate, so a new value cannot change a neighbour count
// that is still being calculated for this same generation.
void computeNextGeneration(const std::vector<int>& currentGrid,
                           std::vector<int>& nextGrid,
                           int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            int neighbours = countNeighbours(currentGrid, N, row, col);
            int alive = currentGrid[row * N + col];
            nextGrid[row * N + col] = nextCellState(alive, neighbours);
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

// Mixes every cell, in order, into one number. The same function in the CUDA
// program makes the final grids comparable with a single printed value.
unsigned long long computeChecksum(const std::vector<int>& grid) {
    unsigned long long checksum = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        checksum = checksum * CHECKSUM_MULTIPLIER +
                   static_cast<unsigned long long>(grid[i]);
    }
    return checksum;
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

    // One contiguous block per grid. row * N + col is the 2D address.
    // Separate rows (vector<vector<int>>) would add pointer chasing and would
    // make this a weaker baseline against the GPU.
    const size_t cellCount = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<int> currentGrid;
    std::vector<int> nextGrid;

    try {
        currentGrid.resize(cellCount);
        nextGrid.resize(cellCount);
    } catch (const std::bad_alloc&) {
        std::cerr << "Not enough memory for a " << N << " x " << N << " grid.\n";
        return 1;
    }

    initializeGrid(currentGrid, N);

    // Timed section: the 100 generations only.
    // Grid setup above, and the checksum below, are outside this clock.
    const auto start = std::chrono::steady_clock::now();

    for (int generation = 0; generation < iterations; ++generation) {
        computeNextGeneration(currentGrid, nextGrid, N);

        // Double buffering. The vectors exchange their storage in constant time.
        // After the swap, currentGrid is the generation just computed.
        std::swap(currentGrid, nextGrid);
    }

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;

    // currentGrid holds generation `iterations` because the loop swaps once per generation.
    const long long livingCells = countLivingCells(currentGrid);
    const unsigned long long checksum = computeChecksum(currentGrid);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------\n";
    std::cout << "Implementation: CPU\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Execution time (ms): " << elapsed.count() << "\n";
    std::cout << "Living cells: " << livingCells << "\n";
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "----------------------------------------\n";

    return 0;
}
