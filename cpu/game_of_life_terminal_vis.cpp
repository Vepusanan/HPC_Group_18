// Interactive Terminal Visualizer for Conway's Game of Life.
//
// Shows real-time animated simulation in the terminal using ANSI escape codes
// and Unicode block characters. Uses the exact same random generator, seed,
// rules, zero boundaries, and checksum as the CPU and CUDA implementations.

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

constexpr unsigned int RANDOM_SEED = 42;
constexpr int DEFAULT_GRID_SIZE = 64;
constexpr int DEFAULT_ITERATIONS = 50;
constexpr int DEFAULT_DELAY_MS = 60;
constexpr unsigned long long CHECKSUM_MULTIPLIER = 1315423911ull;

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [grid_size] [iterations] [delay_ms]\n"
              << "Defaults: grid_size = " << DEFAULT_GRID_SIZE
              << ", iterations = " << DEFAULT_ITERATIONS
              << ", delay_ms = " << DEFAULT_DELAY_MS << "\n"
              << "Example: " << prog << " 64 100 50\n";
}

int nextRandomCell(unsigned int& state) {
    state = 1664525u * state + 1013904223u;
    return static_cast<int>(state >> 31);
}

void initializeGrid(std::vector<int>& grid, int N) {
    unsigned int state = RANDOM_SEED;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            grid[row * N + col] = nextRandomCell(state);
        }
    }
}

int countNeighbours(const std::vector<int>& grid, int N, int row, int col) {
    int neighbours = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int nr = row + dr;
            int nc = col + dc;
            if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
            neighbours += grid[nr * N + nc];
        }
    }
    return neighbours;
}

int nextCellState(int alive, int neighbours) {
    if (alive == 1 && (neighbours == 2 || neighbours == 3)) return 1;
    if (alive == 0 && neighbours == 3) return 1;
    return 0;
}

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
    long long count = 0;
    for (int cell : grid) count += cell;
    return count;
}

unsigned long long computeChecksum(const std::vector<int>& grid) {
    unsigned long long checksum = 0;
    for (int cell : grid) {
        checksum = checksum * CHECKSUM_MULTIPLIER + static_cast<unsigned long long>(cell);
    }
    return checksum;
}

// Renders the grid using Unicode half-blocks (▀) to display 2 vertical cells per character line.
// Supports viewport windowing for larger grids.
void renderTerminal(const std::vector<int>& grid, int N, int gen, int totalGens,
                    long long living, unsigned long long checksum, int maxRows = 32, int maxCols = 80) {
    int viewRows = std::min(N, maxRows * 2);
    int viewCols = std::min(N, maxCols);

    std::string buffer;
    buffer.reserve(viewRows * viewCols * 8);

    // ANSI: Move cursor to home (top-left) without clearing whole screen to avoid flicker
    buffer += "\033[H";

    // Header HUD
    buffer += "\033[1;36m=== CONWAY'S GAME OF LIFE VISUALIZER ===\033[0m\n";
    buffer += "\033[1;33mGrid:\033[0m " + std::to_string(N) + "x" + std::to_string(N);
    buffer += " | \033[1;33mGen:\033[0m " + std::to_string(gen) + " / " + std::to_string(totalGens);
    buffer += " | \033[1;32mAlive:\033[0m " + std::to_string(living);
    buffer += " | \033[1;35mChecksum:\033[0m " + std::to_string(checksum) + "\n";
    buffer += "\033[90m" + std::string(std::min(viewCols, 70), '-') + "\033[0m\n";

    // Draw grid rows 2 by 2 using top half block:
    // Top cell determines foreground color, bottom cell determines background color
    for (int r = 0; r < viewRows; r += 2) {
        for (int c = 0; c < viewCols; ++c) {
            int top = grid[r * N + c];
            int bottom = (r + 1 < N) ? grid[(r + 1) * N + c] : 0;

            if (top && bottom) {
                buffer += "\033[38;5;48;48;5;48m█\033[0m"; // both alive: bright cyan/green
            } else if (top && !bottom) {
                buffer += "\033[38;5;48;48;5;234m▀\033[0m"; // top alive, bottom dead
            } else if (!top && bottom) {
                buffer += "\033[38;5;234;48;5;48m▀\033[0m"; // top dead, bottom alive
            } else {
                buffer += "\033[38;5;236m ·\033[0m"[0]; // dead cell (subtle dot or blank)
                buffer += " ";
            }
        }
        buffer += "\n";
    }

    if (viewRows < N || viewCols < N) {
        buffer += "\033[90m(Displaying top-left " + std::to_string(viewCols) + "x" +
                  std::to_string(viewRows) + " viewport of " + std::to_string(N) + "x" +
                  std::to_string(N) + " grid)\033[0m\n";
    }

    std::cout << buffer << std::flush;
}

int main(int argc, char* argv[]) {
    int N = DEFAULT_GRID_SIZE;
    int iterations = DEFAULT_ITERATIONS;
    int delayMs = DEFAULT_DELAY_MS;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) iterations = std::atoi(argv[2]);
    if (argc >= 4) delayMs = std::atoi(argv[3]);

    if (N <= 0 || iterations <= 0 || delayMs < 0) {
        printUsage(argv[0]);
        return 1;
    }

    std::vector<int> currentGrid(static_cast<size_t>(N) * N);
    std::vector<int> nextGrid(static_cast<size_t>(N) * N);

    initializeGrid(currentGrid, N);

    // Hide terminal cursor & clear screen once
    std::cout << "\033[?25l\033[2J\033[H" << std::flush;

    long long living = countLivingCells(currentGrid);
    unsigned long long checksum = computeChecksum(currentGrid);
    renderTerminal(currentGrid, N, 0, iterations, living, checksum);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs * 2));

    const auto startTime = std::chrono::steady_clock::now();

    for (int gen = 1; gen <= iterations; ++gen) {
        computeNextGeneration(currentGrid, nextGrid, N);
        std::swap(currentGrid, nextGrid);

        living = countLivingCells(currentGrid);
        checksum = computeChecksum(currentGrid);

        renderTerminal(currentGrid, N, gen, iterations, living, checksum);

        if (delayMs > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
    }

    const auto endTime = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = endTime - startTime;

    // Show cursor again
    std::cout << "\033[?25h";

    std::cout << "\n----------------------------------------\n";
    std::cout << "Simulation Complete!\n";
    std::cout << "Grid size: " << N << " x " << N << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Execution time (excluding delay): " << elapsed.count() << " ms\n";
    std::cout << "Living cells: " << living << "\n";
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "----------------------------------------\n";

    return 0;
}
