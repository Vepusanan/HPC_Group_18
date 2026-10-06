# Conway’s Game of Life — Mac CPU vs Colab CUDA

A small experiment: **100 generations, five grid sizes, three runs per size**.
The CPU runs on your Mac. CUDA runs on an NVIDIA GPU in Google Colab.
The dashboard compares these different systems; this is not an isolated measure
of GPU parallelism on identical hardware.

## 1. Run the CPU benchmark on your Mac

Open Terminal in this project folder:

```bash
cd /Users/vepusanan/Desktop/HPC_Group_18
python3 run_benchmarks.py
```

If a compiler is missing, run `xcode-select --install`, finish installation, and
retry. Python 3 is required. No Python packages are needed.

The runner compiles with `-O3`, runs correctness tests, benchmarks
256, 512, 1024, 2048 and 4096 squared, and saves **results/cpu_results.json**.
Every measured sample starts from seed 42 and runs exactly 100 generations.
Close heavy applications before benchmarking. Keep all samples, including slow
ones; variability is displayed rather than silently filtered.

## 2. Open the dashboard

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **http://127.0.0.1:8080/visualizer/**. Serve from the project root, not the
visualizer directory. CPU results load automatically. No npm install, external
chart library, backend or account is needed. Stop the server with Control+C.
If port 8080 is occupied, use 8081 and change the URL accordingly.

## 3. Run CUDA in Colab

Upload **notebooks/Conway_Game_of_Life_HPC_Colab.ipynb** to Google Colab.
Select **Runtime → Change runtime type → GPU**, connect, then **Run all**.
Download **cuda_results.json** from the final cell.

The notebook contains the source code and runner. Its CPU execution is used only
for exact CUDA correctness checks; it does not replace your Mac CPU measurements.
Do not configure CUDA locally on macOS. See [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md).

## 4. Compare

Click **Upload CUDA JSON** in the dashboard and select the downloaded file.
Charts, speedups, the comparison table and measured insights update immediately.
Use **Download comparison CSV** to save the joined data. Save the original CUDA
JSON in `results/` for your submission too. Uploaded data stays in the page's
memory; after reloading, upload CUDA again. CPU data reloads from disk automatically.

The importer accepts only the current schema. It checks iterations, seed, rules,
boundaries, timings, repetitions and sizes. Overlapping grid sizes are paired;
unmatched sizes remain visible without a speedup. A final-grid SHA-256 mismatch
excludes that pair from speedup calculations. Matching hashes are strong evidence
of cross-machine consistency, not a byte-for-byte comparison inside the browser.
The CUDA runner separately checks every output byte against its CPU reference.

## Active files

```text
run_benchmarks.py               # CPU by default; --cuda in Colab
cpu/game_of_life_cpu.cpp        # Contiguous logical 2D array, double buffering
cuda/game_of_life_cuda.cu       # Baseline 16×16 CUDA blocks
notebooks/                     # Self-contained CUDA notebook
results/cpu_results.json        # Measured Mac CPU results
results/cuda_results.json       # Downloaded CUDA results (when available)
visualizer/                    # HTML, CSS, JS; local SVG charts
README.md / COLAB_INSTRUCTIONS.md
REPORT.md / PRESENTATION_SCRIPT.md
```

`tests/` holds CPU tests. `scripts/build_colab_notebook.py` is only for maintainers:
after editing the runner or C++ files, run it to update the notebook. Ordinary
execution needs neither the builder nor manual source uploads.

## Measurements and correctness

- CPU: monotonic host-clock interval around 100 updates.
- CUDA: event interval around 100 launches, including device timeline gaps.
- CUDA total: host-clock H2D + simulation + synchronized D2H.
- Allocation, initialization, warm-up, validation, printing and export are excluded.
- CUDA warms the kernel once in each process and restores the input before timing.
  CPU measurements retain ordinary initial cache effects; document this difference.
- Both versions use the same LCG initializer, seed 42, B3/S23 and dead exterior.
- Three samples are averaged; sample standard deviation is retained.
- Speedup = mean CPU simulation time / mean CUDA simulation time. Total speedup
  uses CUDA total time as denominator. Ratios describe the measured systems only.
- Export occurs only after the entire run succeeds. Existing output survives a
  failed run, so check timestamps before submitting.

CPU tests:

```bash
mkdir -p build
clang++ -std=c++17 -O1 -g -fsanitize=address,undefined tests/test_cpu.cpp -o build/test_cpu
./build/test_cpu
node tests/test_dashboard.cjs
```

Node is optional and needed only for the dashboard development test. The browser
itself requires no Node installation. CUDA verification is pending until Colab runs.

## Deliverables

Submit the source, executed notebook, both original JSON files, comparison CSV,
report, chart screenshots and a video of at most three minutes. Record the actual
Mac CPU, Colab GPU, compiler versions and timing definitions from the JSON files.
Do not infer GPU speedups before uploading real CUDA results. Optional optimizations are not required for this baseline project.
