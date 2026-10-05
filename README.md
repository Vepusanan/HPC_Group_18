# Conway's Game of Life: High Performance Computing Benchmark (CPU vs. CUDA)
### Assignment 1 — High Performance Computing (Group 18)

An empirical High Performance Computing investigation comparing sequential CPU execution against massively parallel GPU acceleration using NVIDIA CUDA for Conway's Game of Life.

Both implementations are benchmarked in an identical **Google Colab environment** (NVIDIA Tesla T4 GPU + Intel Xeon CPU) to ensure fairness, consistency, and eliminate cross-architecture hardware distortion.

---

## Repository Structure

```text
.
├── cpu/
│   ├── game_of_life_cpu.cpp          # Optimized sequential C++ implementation (-O3, contiguous 1D array)
│   └── game_of_life_terminal_vis.cpp # Optional ANSI terminal visualizer
├── cuda/
│   ├── game_of_life_cuda.cu          # Baseline CUDA implementation (16x16 thread blocks, coalesced loads)
│   └── game_of_life_cuda_shared.cu   # Optional shared-memory tiled CUDA implementation (18x18 halo tile)
├── notebooks/
│   └── Conway_Game_of_Life_HPC_Colab.ipynb # Pre-configured, ready-to-run Google Colab Notebook
├── results/
│   ├── benchmark_results.json        # Structured JSON benchmark results (consumed by visualizer)
│   ├── benchmark_results.csv         # Complete CSV benchmark telemetry
│   └── results.csv                   # Summary CSV for report
├── visualizer/                       # Interactive HPC Performance Dashboard & Simulation
│   ├── index.html                    # Single-page engineering dashboard
│   ├── style.css                     # Glassmorphic responsive styling & themes
│   └── app.js                        # Canvas simulation, Chart.js graphs, dynamic insights
├── COLAB_INSTRUCTIONS.md             # Step-by-step Google Colab copy-paste instructions
├── REPORT.md                         # Comprehensive academic technical report (12 sections)
├── PRESENTATION_SCRIPT.md            # 3-minute video presentation script with exact timestamps
├── run_benchmarks.py                 # Automated benchmark harness with multi-run averaging
└── README.md                         # Project documentation
```

---

## Core HPC Features & Design Decisions

### 1. Sequential CPU Implementation (`cpu/game_of_life_cpu.cpp`)
- **Contiguous 1D Allocation:** Uses row-major index mapping `index = row * N + col` in a single contiguous memory block (`std::vector<int>`), maximizing L1/L2 cacheline prefetching and eliminating pointer chasing.
- **Double Buffering:** Maintains `currentGrid` and `nextGrid`, swapping pointers in $O(1)$ time via `std::swap` to eliminate redundant memory copying between generations.
- **Dead Borders:** Strictly implements zero boundary conditions (cells outside $N \times N$ evaluate to dead).
- **Isolated Timing:** Timed strictly around the 100 simulation iterations using `std::chrono::high_resolution_clock`.

### 2. Massively Parallel CUDA Implementation (`cuda/game_of_life_cuda.cu`)
- **1 Thread per Cell:** Maps each cell to a unique CUDA thread: `col = blockIdx.x * blockDim.x + threadIdx.x`.
- **$16 \times 16$ 2D Thread Blocks:** 256 threads per block (8 warps). Consecutive threads access adjacent columns, guaranteeing **100% coalesced 128-byte DRAM transactions**.
- **Device-Resident Memory:** Grids remain in GPU VRAM across all 100 iterations. Only device pointers swap on the host in $O(1)$ time.
- **Dual Timing Instrumentation:** Measures pure CUDA kernel time (100 iterations) using `cudaEvent_t`, as well as total end-to-end time ($H \to D$ + 100 iterations + $D \to H$).

### 3. Verification & Bitwise Correctness
Both programs use an identical Linear Congruential PRNG (Seed 42) and compute:
1. Total living cell count after 100 iterations.
2. A 64-bit cumulative polynomial checksum (`checksum = checksum * 1315423911 + cell`).
- Both programs yield **100% identical checksums and living cell counts (0 mismatched cells)** across all evaluated grid sizes.

---

## Benchmark Results (Google Colab — NVIDIA Tesla T4 vs Intel Xeon)

| Grid Size | Total Cells ($N^2$) | Iterations | CPU Time (ms) | CUDA Kernel (ms) | CUDA Total (ms) | Kernel Speedup | Total Speedup | Verified Living | 64-Bit Checksum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$256 \times 256$** | 65,536 | 100 | **21.82** | **1.12** | 2.45 | **$19.47\times$** | $8.90\times$ | 5,877 | `2188031159639976069` |
| **$512 \times 512$** | 262,144 | 100 | **86.42** | **1.84** | 4.10 | **$46.96\times$** | $21.07\times$ | 23,852 | `3838351066650152210` |
| **$1024 \times 1024$** | 1,048,576 | 100 | **345.18** | **4.62** | 10.85 | **$74.71\times$** | $31.82\times$ | 99,296 | `14789994132222743192` |
| **$2048 \times 2048$** | 4,194,304 | 100 | **1,418.52** | **15.20** | 38.60 | **$93.30\times$** | $36.75\times$ | 390,059 | `17074688330164608745` |
| **$4096 \times 4096$** | 16,777,216 | 100 | **6,854.21** | **58.40** | 139.22 | **$117.36\times$** | **$49.23\times$** | 1,584,269 | `15231916768216565527` |

---

## How to Run on Google Colab

See [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md) for detailed step-by-step instructions.

### Quick Start:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `notebooks/Conway_Game_of_Life_HPC_Colab.ipynb`.
3. Set runtime to **GPU (T4 GPU)** via `Runtime` → `Change runtime type`.
4. Run all cells (`Runtime` → `Run all`).
5. Download `benchmark_results.json` and `benchmark_results.csv` to update the dashboard.

---

## Interactive HPC Dashboard & Visualizer

Open `visualizer/index.html` in any web browser, or launch a local web server:
```bash
python3 -m http.server 8080 --directory visualizer
```
Navigate to `http://localhost:8080`.

**Dashboard Highlights:**
- **Executive KPI Banner:** Peak speedup ($117.4\times$), CPU time, CUDA time, and validation badges.
- **Interactive Canvas Visualizer:** Seed 42 PRNG, custom drawing, pattern stamping (Glider, Pulsar, Gosper Gun, Acorn), and live checksum telemetry.
- **Architecture Visualizer:** CPU serial instruction flow vs. GPU 2D thread block coalesced architecture.
- **Interactive Performance Charts:** Logarithmic execution time curves and speedup bar charts powered by Chart.js.
- **Dynamic Insights:** Automated analysis of Amdahl's Law, PCIe latency, and $230\text{ GB/s}$ sustained memory bandwidth saturation.
- **Custom Upload:** Upload custom Colab benchmark JSON or CSV files to visualize live results instantly.

---

## Documentation & Deliverables

- **Full Academic Technical Report:** [`REPORT.md`](REPORT.md) (12 sections covering theory, implementation, profiling, Amdahl's law, and optimizations).
- **3-Minute Presentation Script:** [`PRESENTATION_SCRIPT.md`](PRESENTATION_SCRIPT.md) (exact timestamped narration script for video recording).
- **Colab Execution Guide:** [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md).
