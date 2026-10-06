# CUDA in Colab — three steps

1. Open https://colab.research.google.com and upload
   `notebooks/Conway_Game_of_Life_HPC_Colab.ipynb`.
2. Select **Runtime → Change runtime type → GPU**, save and connect.
3. Choose **Runtime → Run all**. Download `cuda_results.json` when prompted.

Then open your local dashboard and click **Upload CUDA JSON**. No benchmark
values need to be entered manually.

## What the cells do, in order

1. `!nvidia-smi` and `!nvcc --version` confirm the GPU and compiler.
2. Create working directories.
3. Write the embedded CPU source, CUDA source, CPU tests and runner.
4. Run `python3 run_benchmarks.py --cuda`.
5. Download `results/cuda_results.json`.

The runner compiles CPU reference code with `g++ -O3 -std=c++17` and CUDA with
`nvcc -O3 -std=c++17`. It checks small grids and partial blocks, then compares each
measured CUDA result byte-for-byte with a CPU reference. The CPU reference is
correctness work only. The dashboard uses your **Mac CPU dataset** for speedup.

Both datasets use sizes 256, 512, 1024, 2048 and 4096, 100 generations, seed 42,
and three timing samples. If you change `SIZES` in the runner, regenerate the
notebook with `python3 scripts/build_colab_notebook.py` and rerun both environments.

Expected final output: `Saved .../results/cuda_results.json`.
If validation or compilation fails, no new JSON is exported. Read the cell error;
do not use an old file as evidence of a successful run. A busy cell can remain
quiet while a large validation or benchmark process runs.

If no GPU is allocated, reconnect to a GPU runtime when capacity is available.
Do not install CUDA on the Mac. Download the JSON and executed notebook before
ending the Colab session. Save the JSON locally under `results/cuda_results.json`.

CUDA JSON contains simulation and total timing. Total includes transfers and
synchronization but excludes context creation, allocations and warm-up. Keep
these scopes in the report. The Mac and Colab are different machines.
