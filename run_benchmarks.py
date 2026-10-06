#!/usr/bin/env python3
"""CPU: python3 run_benchmarks.py; Colab GPU: python3 run_benchmarks.py --cuda."""
import argparse
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


def timing(data, name):
    value = float(data[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid timing: ' + name)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='Run on a Colab NVIDIA GPU')
    args = parser.parse_args()
    kind = 'CUDA' if args.cuda else 'CPU'
    compiler = shutil.which('clang++') if platform.system() == 'Darwin' else shutil.which('g++')
    if not compiler:
        raise SystemExit('C++ compiler missing. On Mac: xcode-select --install')
    if args.cuda and not shutil.which('nvcc'):
        raise SystemExit('CUDA requires a Colab NVIDIA GPU runtime. Do not install CUDA on your Mac.')
    hardware = command(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']) if args.cuda else (
        command(['sysctl', '-n', 'machdep.cpu.brand_string']) if platform.system() == 'Darwin' else command(['lscpu']))
    source = 'cuda/game_of_life_cuda.cu' if args.cuda else 'cpu/game_of_life_cpu.cpp'
    data = dict(schemaVersion=1, implementation=kind, iterations=ITERATIONS,
                gridSizes=SIZES, runs=RUNS, seed=42, boundary='dead-exterior',
                initializer='lcg32-msb-v1', rule='B3/S23',
                createdAt=datetime.now(timezone.utc).isoformat(),
                environment=dict(hardware=hardware, platform=platform.platform(),
                                 compiler=command(['nvcc' if args.cuda else compiler, '--version']),
                                 flags='-O3 -std=c++17'),
                timingScope='100 simulation iterations; excludes allocation, initialization, validation and export',
                sourceSHA256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest(), results=[])
    if args.cuda:
        data['totalTimingScope'] = 'H2D + simulation + synchronized D2H; excludes allocation and warm-up'
        data['blockSize'] = [16, 16]
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        cpu, gpu, test = temp/'cpu', temp/'gpu', temp/'test'
        command([compiler, '-O3', '-std=c++17', 'tests/test_cpu.cpp', '-o', test])
        print(command([test]), flush=True)
        command([compiler, '-O3', '-std=c++17', 'cpu/game_of_life_cpu.cpp', '-o', cpu])
        if args.cuda:
            command(['nvcc', '-O3', '-std=c++17', source, '-o', gpu])
            # Correctness only: the Colab CPU is NOT the measured CPU dataset.
            for n in [1, 2, 3, 15, 16, 17, 31, 33]:
                for iterations in [1, 2, 100]:
                    command([cpu, n, iterations, temp/'reference'])
                    command([gpu, n, iterations, temp/'actual'])
                    if (temp/'reference').read_bytes() != (temp/'actual').read_bytes():
                        raise RuntimeError(f'CUDA validation failed: {n}, {iterations}')
        for n in SIZES:
            samples, totals, digest = [], [], None
            if args.cuda:
                command([cpu, n, ITERATIONS, temp/'reference'])
                reference = (temp/'reference').read_bytes()
            for repetition in range(RUNS):
                result = fields(command([gpu if args.cuda else cpu, n, ITERATIONS, temp/'actual']))
                grid = (temp/'actual').read_bytes()
                if len(grid) != n*n or any(cell > 1 for cell in grid):
                    raise RuntimeError('Invalid grid output')
                if args.cuda and grid != reference:
                    raise RuntimeError(f'CUDA validation failed at {n}')
                current = hashlib.sha256(grid).hexdigest()
                if digest is not None and digest != current:
                    raise RuntimeError('Non-deterministic output')
                digest = current
                samples.append(timing(result, 'Kernel execution time (ms)' if args.cuda else 'Execution time (ms)'))
                if args.cuda:
                    totals.append(timing(result, 'Total GPU time (inc. transfers) (ms)'))
                print(f'{kind} {n}x{n}, run {repetition+1}/{RUNS}: {samples[-1]:.3f} ms', flush=True)
            row = dict(gridSize=n, cells=n*n, executionTimeMs=statistics.mean(samples),
                       stddevMs=statistics.stdev(samples), samplesMs=samples,
                       finalGridSHA256=digest, livingCells=int(result['Living cells']),
                       validation='exact-cpu-reference' if args.cuda else 'cpu-tests-and-repeatability')
            if totals:
                row.update(totalTimeMs=statistics.mean(totals), totalSamplesMs=totals)
            data['results'].append(row)
    destination = ROOT/'results'
    destination.mkdir(exist_ok=True)
    path = destination/f'{kind.lower()}_results.json'
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(data, indent=2)+'\n')
    pending.replace(path)
    print(f'Saved {path}', flush=True)
    if not args.cuda:
        print('Next: python3 -m http.server 8080 --bind 127.0.0.1\nOpen http://127.0.0.1:8080/visualizer/')


if __name__ == '__main__':
    main()
