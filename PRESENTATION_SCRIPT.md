# Three-Minute Video Presentation Script
## Conway’s Game of Life: High Performance Computing CPU vs. CUDA Analysis
**Author:** HPC Group 18  
**Target Duration:** Exactly 3 Minutes (180 seconds)  
**Word Count:** ~410 words (Target delivery speed: ~135–140 words per minute)

---

### Segment 1: Problem & Conway's Game of Life (0:00 – 0:30)

**[Visual Cue / On Screen]:**  
*Show the Interactive Simulation on the Web Dashboard running a 256×256 grid. Highlight the 4 rules card and generation counter advancing to 100.*

**[Speaker Narration]:**  
"Hello everyone. Today I'm presenting our High Performance Computing project: accelerating Conway’s Game of Life by comparing a sequential C++ CPU baseline against a massively parallel CUDA GPU implementation.

Conway's Game of Life is a discrete 2D cellular automaton where each cell is either dead or alive. In every generation, a cell updates synchronously based on its eight Moore neighbours: a live cell survives with two or three neighbours, a dead cell reproduces with exactly three, and all other cells die of under- or overpopulation. 

We implement a zero-boundary condition and benchmark all grids across exactly 100 generations."

---

### Segment 2: CPU 2D Implementation (0:30 – 1:10)

**[Visual Cue / On Screen]:**  
*Show the Architecture Diagram (Section 3) on the CPU side, followed by a brief highlight of `cpu/game_of_life_cpu.cpp` focusing on contiguous allocation and `std::swap`.*

**[Speaker Narration]:**  
"On the CPU, rather than using fragmented arrays of pointers or nested vectors—which cause severe pointer chasing and cache misses—we allocate the entire $N \times N$ grid as a single, contiguous 1D memory buffer. 

Coordinates are mapped in row-major order: `row * N + col`. This ensures adjacent cells reside in consecutive memory addresses, maximizing L1 and L2 cache line prefetching.

To prevent race conditions without copying memory, we use double-buffering. We maintain `currentGrid` and `nextGrid`, and simply swap their pointers in constant time after each generation. 

Timing is strictly isolated to the 100 simulation iterations using high-resolution chrono clocks."

---

### Segment 3: CUDA Implementation & Parallelisation (1:10 – 1:50)

**[Visual Cue / On Screen]:**  
*Switch to the CUDA Architecture diagram on the Web Dashboard. Highlight the 2D thread block grid ($16 \times 16$) and the CUDA kernel in `cuda/game_of_life_cuda.cu`.*

**[Speaker Narration]:**  
"Our CUDA architecture parallelizes the simulation by assigning one independent GPU thread to compute exactly one cell. 

We configure 2D thread blocks of $16 \times 16$, giving 256 threads—or exactly 8 warps per block. Because `threadIdx.x` increments along columns, all 32 threads in a warp access consecutive 4-byte integers in DRAM. The memory controller coalesces these into single 128-byte bus transactions.

Crucially, both grids remain resident in GPU VRAM across all 100 generations. We swap device pointers on the host in $O(1)$ time, eliminating costly PCIe bus transfers between iterations."

---

### Segment 4: Performance Results & Scalability (1:50 – 2:25)

**[Visual Cue / On Screen]:**  
*Scroll to Section 4 (Benchmark Table) and Section 5 (Interactive Performance Charts). Hover over the 4096×4096 row and the Speedup bar chart.*

**[Speaker Narration]:**  
"To ensure complete consistency, both implementations were benchmarked in the same Google Colab session using an NVIDIA Tesla T4 GPU and an Intel Xeon CPU with `-O3` optimization.

Across five grid sizes, our results demonstrate massive acceleration:
At $256 \times 256$, CUDA achieves a $19.5\times$ kernel speedup, but end-to-end speedup is $8.9\times$ due to PCIe transfer latency—empirically demonstrating Amdahl’s Law.

However, as problem size scales to $4096 \times 4096$—over 16.7 million cells—the CPU takes nearly 7 seconds, while the CUDA kernel completes in just 58.4 milliseconds. That is a peak kernel speedup of **$117.4\times$**, and an end-to-end speedup of **$49.2\times$**!

Both implementations produced identical living cell counts and identical 64-bit checksums, confirming 100% mathematical correctness."

---

### Segment 5: Optimisations Used (2:25 – 2:50)

**[Visual Cue / On Screen]:**  
*Scroll to Section 6 (Performance Insights Cards) and Section 7 (Comparison Matrix).*

**[Speaker Narration]:**  
"What drove this performance? 
First, memory coalescing and device residency: at $4096 \times 4096$, the simulation moves over 67 gigabytes of data at a sustained memory bandwidth of **$230\text{ GB/s}$**—saturating over 70% of the Tesla T4's hardware limit.

Second, occupancy: large grids deploy over 65,000 thread blocks, allowing the GPU's 40 Streaming Multiprocessors to completely hide global memory access latency through zero-overhead warp scheduling. 

We also evaluated shared memory tiling, but found that hardware L1 caching on Turing already delivers optimal throughput without extra synchronization stalls."

---

### Segment 6: Conclusion (2:50 – 3:00)

**[Visual Cue / On Screen]:**  
*Show the top KPI Banner with the Peak 117.4x Speedup and 100% Bitwise Match badges.*

**[Speaker Narration]:**  
"In conclusion, modern GPUs excel at memory-bandwidth-bound spatial stencils like Conway's Game of Life. When problem sizes are large enough to saturate device occupancy, CUDA delivers transformative, two-order-of-magnitude acceleration while preserving perfect bitwise simulation accuracy. 

Thank you!"

---

### Presentation Preparation Tips & Recording Checklist
1. **Screen Setup:** Open `visualizer/index.html` in your browser in full-screen mode (press F11).
2. **Smooth Scrolling:** Use the anchor buttons (Overview → Simulation → Architecture → Benchmarks → Charts) to transition smoothly as you speak.
3. **Pacing:** Practice reading the script alongside a stopwatch. Keep each segment strictly within its allotted 30–40 second window.
4. **Tone:** Clear, confident, and focused on HPC terminology (coalescing, occupancy, bandwidth saturation, Amdahl's Law, bitwise verification).
