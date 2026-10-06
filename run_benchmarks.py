#!/usr/bin/env python3
"""Run on the Colab GPU runtime: exact validation, then 100-generation benchmarks."""
import csv
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIZES = [256, 512, 1024, 2048, 4096]
ITERATIONS = 100
RUNS = 3


def command(args):
    return subprocess.run([str(x) for x in args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def fields(output):
    return dict(line.split(': ', 1) for line in output.splitlines() if ': ' in line)


def timing(values, key):
    value = float(values[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid measurement: {key}={value}')
    return value


def validate(cpu_path, gpu_path, cells):
    a, b = cpu_path.read_bytes(), gpu_path.read_bytes()
    if len(a) != cells or len(b) != cells:
        raise RuntimeError('Incomplete grid output')
    mismatches = sum(x != y for x, y in zip(a, b))
    if mismatches:
        raise RuntimeError(f'Validation: FAILED ({mismatches} mismatched cells)')
    print(f'Validation: PASSED ({cells} cells, 0 mismatches)', flush=True)


def main():
    if platform.system() != 'Linux' or not shutil.which('nvcc'):
        raise SystemExit('Run this comparison in a Linux NVIDIA GPU runtime (Google Colab). Existing results were not changed.')
    gpu_info = command(['nvidia-smi'])
    environment = {
        'timestampUTC': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'cpu': command(['lscpu']),
        'gpu': gpu_info, 'cudaCompiler': command(['nvcc', '--version']),
        'cpuCompiler': command(['g++', '--version']),
        'cpuFlags': '-O3 -std=c++17', 'cudaFlags': '-O3 -std=c++17',
        'gridSizes': SIZES, 'iterations': ITERATIONS, 'runs': RUNS,
        'seed': 42, 'boundary': 'outside-domain cells dead',
        'cpuThreads': 1, 'blockSize': [16, 16],
        'validation': 'exact byte-per-cell comparison before timing sweep and every repetition',
        'gpuTotalTiming': 'steady_clock: H2D + simulation + synchronized D2H; excludes allocation/context/warm-up',
        'gpuSimulationTiming': 'CUDA events around 100 launches; includes device timeline gaps between launches',
        'sourceSHA256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT/'cpu/game_of_life_cpu.cpp', ROOT/'cuda/game_of_life_cuda.cu']}
    }
    # Temporary binaries avoid accidentally executing stale binaries after a compile failure.
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        cpu, gpu, tests = [temp / name for name in ('cpu', 'gpu', 'tests')]
        command(['g++', '-O3', '-std=c++17', 'tests/test_cpu.cpp', '-o', tests])
        print(command([tests]), flush=True)
        command(['g++', '-O3', '-std=c++17', 'cpu/game_of_life_cpu.cpp', '-o', cpu])
        command(['nvcc', '-O3', '-std=c++17', 'cuda/game_of_life_cuda.cu', '-o', gpu])
        a, b = temp/'cpu.bin', temp/'gpu.bin'
        # Odd/even iterations exercise pointer parity; nonmultiples exercise partial blocks.
        for size in [1, 2, 3, 15, 16, 17, 31, 33, 65]:
            for generations in [1, 2, 100]:
                command([cpu, size, generations, a])
                command([gpu, size, generations, b])
                validate(a, b, size*size)
        # Verify all assignment sizes before starting the measured sweep.
        for size in SIZES:
            command([cpu, size, ITERATIONS, a])
            command([gpu, size, ITERATIONS, b])
            validate(a, b, size*size)
        records = []
        for size in SIZES:
            samples = []
            for repetition in range(RUNS):
                outputs = {}
                # Alternate execution order to reduce systematic ordering bias.
                order = [('cpu', cpu, a), ('gpu', gpu, b)]
                if repetition % 2: order.reverse()
                for name, executable, destination in order:
                    outputs[name] = fields(command([executable, size, ITERATIONS, destination]))
                validate(a, b, size*size)
                c, g = outputs['cpu'], outputs['gpu']
                samples.append({'cpuTimeMs': timing(c, 'Execution time (ms)'),
                                'gpuKernelTimeMs': timing(g, 'Kernel execution time (ms)'),
                                'gpuTotalTimeMs': timing(g, 'Total GPU time (inc. transfers) (ms)')})
            means = {key: statistics.mean(row[key] for row in samples) for key in samples[0]}
            record = {'gridSize': size, 'cells': size*size, 'iterations': ITERATIONS,
                      'runs': RUNS, 'blockSize': '16x16', **means,
                      'speedup': means['cpuTimeMs']/means['gpuKernelTimeMs'],
                      'speedupTotal': means['cpuTimeMs']/means['gpuTotalTimeMs'],
                      'validation': 'PASSED', 'mismatchedCells': 0,
                      'gpuName': g['GPU name'], 'checksum': c['Checksum'],
                      'livingCells': int(c['Living cells']), 'samples': samples,
                      'stddevMs': {key: statistics.stdev(row[key] for row in samples) for key in means}}
            records.append(record)
            print(f'{size}x{size}: CPU {means["cpuTimeMs"]:.3f} ms, CUDA {means["gpuKernelTimeMs"]:.3f} ms, speedup {record["speedup"]:.2f}x', flush=True)
    # Export only after the entire sweep succeeds. No partial or CPU-only comparison.
    destination = ROOT/'results'
    destination.mkdir(exist_ok=True)
    (destination/'benchmark_results.json').write_text(json.dumps(records, indent=2)+'\n')
    (destination/'benchmark_environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    columns = [key for key in records[0] if key not in ('samples', 'stddevMs')]
    with (destination/'benchmark_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print('Exported validated results and environment metadata to results/.')


if __name__ == '__main__':
    main()
