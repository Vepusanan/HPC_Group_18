"""Maintainer command: embed current sources in a self-contained CUDA notebook."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
cells = []
def add(kind, text):
    cell = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code': cell.update(execution_count=None, outputs=[])
    cells.append(cell)
add('markdown', '# CUDA Game of Life benchmark\nSelect Runtime → Change runtime type → GPU, then Run all. Download cuda_results.json and upload it to the Mac dashboard. The Colab CPU is used only to check correctness; comparison CPU timings come from your Mac.\n')
add('code', '!nvidia-smi\n!nvcc --version\n')
add('code', 'from pathlib import Path\nfor directory in ["cpu", "cuda", "tests", "results"]:\n    Path(directory).mkdir(exist_ok=True)\n')
for name in ['cpu/game_of_life_cpu.cpp','cuda/game_of_life_cuda.cu','tests/test_cpu.cpp','run_benchmarks.py']:
    add('code', '%%writefile '+name+'\n'+(root/name).read_text())
add('code', 'import subprocess\nsubprocess.run(["python3", "run_benchmarks.py", "--cuda"], check=True)\n')
add('code', 'from google.colab import files\nfiles.download("results/cuda_results.json")\n')
nb=dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU'}, nbformat=4, nbformat_minor=4)
(root/'notebooks/Conway_Game_of_Life_HPC_Colab.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
