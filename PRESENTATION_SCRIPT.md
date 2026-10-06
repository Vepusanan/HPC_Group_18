# Three-minute presentation script

**Topic:** Conway’s Game of Life — local CPU versus Colab CUDA  
**Audience:** High Performance Computing assignment assessment  
**Version:** Current measured CPU results; CUDA execution pending

The narration below is suitable for the project's current status. It contains no invented GPU measurements. Read only the quoted narration; screen directions are not spoken. Aim to finish in 2:50–2:58 and verify the recorded duration. Rehearsal, rather than word count alone, determines whether the video meets the three-minute limit.

## 0:00–0:30 — Problem and rules

**Show:** Dashboard title, then briefly start and pause the Game of Life animation.

> Our project compares CPU and CUDA implementations of Conway’s Game of Life. Each cell is alive or dead and considers eight neighbours. A live cell survives with two or three neighbours; otherwise it dies. A dead cell becomes alive with exactly three. All cells update simultaneously. We test five grid sizes for exactly one hundred generations.

## 0:30–1:10 — CPU implementation and experiment

**Show:** `updateGrid()` and `runSimulation()` in the CPU source, then the CPU results table.

> The CPU implementation runs locally on an Apple M1 Mac. It stores the two-dimensional grid in contiguous memory and processes cells using nested loops. One buffer holds the current generation and another receives the next. Swapping pointers avoids copying the grid. Positions outside the domain count as dead. We use seed forty-two and three repetitions per size. A monotonic clock measures only the simulation, excluding initialization and printing. The runner saves the actual samples and averages as JSON.

## 1:10–1:50 — CUDA implementation and validation

**Show:** CUDA thread-coordinate calculation, 16 × 16 launch configuration and Colab notebook.

> The CUDA implementation is prepared for an NVIDIA GPU in Google Colab. Each valid thread updates one cell, using sixteen-by-sixteen thread blocks. Both buffers remain on the GPU for all one hundred generations. The measured run transfers input once and returns the final grid once. CUDA events measure simulation time; a host clock also measures computation plus transfers. The notebook checks CUDA output against a CPU reference. That Colab CPU is used for correctness, not our performance comparison.

## 1:50–2:25 — Measured results and graphs

**Show:** CPU time-growth chart and the empty CUDA comparison state.

> Our measured CPU averages range from about twenty-three milliseconds at two hundred and fifty-six squared to five-point-three-seven seconds at four thousand and ninety-six squared. Larger grids show roughly fourfold time growth when the cell count quadruples. Some samples vary substantially, so we retain every run. CUDA measurements are still pending. Uploading the completed CUDA JSON will calculate CPU time divided by GPU time. Those ratios will compare different machines and compilers.

## 2:25–2:50 — Performance choices

**Show:** Buffer swapping and the dashboard’s implementation explanation.

> We use contiguous arrays, preallocated buffers, pointer swapping and compiler optimization. CUDA also avoids transfers between generations. These reduce avoidable work, but we have not measured each optimization separately. The browser animation demonstrates the rules; it is not a CUDA benchmark. CPU correctness tests have passed, and the notebook will check GPU results before export.

## 2:50–3:00 — Conclusion

**Show:** Results table and project title.

> The CPU results establish our baseline. Validated Colab measurements are the remaining step before we can conclude where CUDA provides an advantage.

---

## Update after the actual CUDA run

Before recording a final completed-experiment presentation:

1. Run the current notebook, retain its validation output and download the CUDA JSON.
2. Upload it to the dashboard and confirm matching grid fingerprints.
3. Replace the results narration with the measured comparison below. Fill every bracket from the same current datasets; do not read the brackets aloud.
4. Change “is prepared for” to “runs on” in the CUDA section. Change “will check GPU results” to “checked GPU results” only after those checks pass.
5. Replace the closing sentence with one supported finding. Rehearse again and stay below three minutes.

**Replacement results narration, 1:50–2:25:**

> Both implementations completed one hundred generations at every tested size. At [selected grid], the CPU averaged [CPU milliseconds] and CUDA simulation averaged [CUDA milliseconds], giving [simulation speedup] times speedup. Including transfers, the speedup was [total speedup]. The highest simulation speedup occurred at [actual peak grid]. The graph shows [observed scaling trend]. These results compare our Mac CPU with the assigned Colab GPU, so hardware and compiler differences contribute to the outcome.

**Replacement conclusion, 2:50–3:00:**

> For our tested configurations, [one measured finding]. Exact validation in Colab and matching result fingerprints support the correctness of the comparison.

## Recording checklist

- [ ] Use the same measurements in the report, dashboard and narration.
- [ ] Present the current-status script if CUDA remains unmeasured; do not describe the experiment as complete.
- [ ] For the completed experiment, replace every bracket and remove outdated pending statements.
- [ ] Show readable source and chart labels; avoid scrolling through long files.
- [ ] Distinguish simulation time from transfer-inclusive time.
- [ ] Record a short microphone test before the full take.
- [ ] Replay the finished video and confirm its duration is no more than 3:00.
