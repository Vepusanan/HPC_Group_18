"""Keep the executable notebook and copyable cells synchronized with source files."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
cells = []
def cell(kind, source):
    entry = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code': entry.update(execution_count=None, outputs=[])
    cells.append(entry)
cell('markdown', '''# Conway's Game of Life: CPU versus CUDA
Select Runtime → Change runtime type → GPU, then run cells in order.
Both executables run in this same Linux GPU runtime; the Mac is only the editor.
The runner validates exact grids before the sweep, runs 100 generations three times
at each of five sizes, and exports means, samples, standard deviations and metadata.
Existing repository results are not evidence for this new run.
''')
cell('code', '!nvidia-smi\n!nvcc --version\n!g++ --version\n!lscpu\n')
cell('code', 'from pathlib import Path\nfor name in ["cpu", "cuda", "tests", "results"]:\n    Path(name).mkdir(exist_ok=True)\n')
for name in ['cpu/game_of_life_cpu.cpp','tests/test_cpu.cpp','cuda/game_of_life_cuda.cu','run_benchmarks.py']:
    cell('code', f'%%writefile {name}\n'+(root/name).read_text())
cell('code', '''import subprocess
subprocess.run(["python3", "run_benchmarks.py"], check=True)
''')
cell('code', '''import json
from pathlib import Path
from google.colab import files
rows = json.loads(Path("results/benchmark_results.json").read_text())
assert len(rows) == 5 and all(r["validation"] == "PASSED" for r in rows)
for name in ["benchmark_results.json", "benchmark_results.csv", "benchmark_environment.json"]:
    files.download("results/" + name)
''')
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
(root/'notebooks/Conway_Game_of_Life_HPC_Colab.ipynb').write_text(json.dumps(notebook, indent=2)+'\n')
text = '# Google Colab execution guide\n\nOpen the notebook in `notebooks/`, or paste these cells into a new Colab notebook in order. Choose a GPU runtime first.\n\n'
for i, item in enumerate(cells,1):
    source=''.join(item['source'])
    text += source+'\n' if item['cell_type']=='markdown' else f'## Cell {i-1}\n\n```python\n{source}\n```\n\n'
text += '''## Timing and interpretation

CPU simulation uses a monotonic host clock. CUDA simulation uses events around
100 kernel launches. CUDA total uses a host clock around the initial copy,
simulation, synchronization and final copy. Allocation, initialization, context
creation, warm-up, validation and file output are excluded. Each CUDA process
warms its kernel before restoring the initial grid. CPU measurements include
ordinary first-iteration cache effects; neither measurement includes compilation.

The 16×16 block contains 256 threads (eight warps). A warp spans two row segments;
this supports adjacent-column accesses but does not guarantee perfect memory
transactions or optimal occupancy. Shared memory is an optional experiment in
`cuda/game_of_life_cuda_shared.cu`, not part of this validated baseline sweep.

Same-session execution improves comparability but Colab load and clock variation
remain uncontrolled. Record the exported CPU, GPU, compiler and CUDA details.
Speedup is the ratio of mean CPU time to mean GPU time, separately for simulation
and total time. Do not infer bandwidth saturation or monotonic scaling without
evidence. The current runner stops on compilation, execution or validation failure.

After the run, return all three exported files for analysis, dashboard integration,
and final report/video revisions. CUDA has not been verified on the Mac.

References: [Game of Life rules](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life),
[NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).
'''
(root/'COLAB_INSTRUCTIONS.md').write_text(text)
