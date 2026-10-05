#!/usr/bin/env python3
"""
Automated HPC Benchmark Suite for Conway's Game of Life
CPU (Contiguous 2D Array) vs CUDA GPU (Global Memory & Tiled)
Author: HPC Group 18
"""

import subprocess
import os
import sys
import json
import statistics
import time

GRID_SIZES = [256, 512, 1024, 2048, 4096]
DEFAULT_ITERATIONS = 100
DEFAULT_REPETITIONS = 3  # Multiple runs per grid size for robust statistical averaging

# Reference living cell counts and 64-bit checksums for 100 iterations (Seed 42)
REFERENCE_CHECKSUMS = {
    256: {"living": 5877, "checksum": "2188031159639976069"},
    512: {"living": 23852, "checksum": "3838351066650152210"},
    1024: {"living": 99296, "checksum": "14789994132222743192"},
    2048: {"living": 390059, "checksum": "17074688330164608745"},
    4096: {"living": 1584269, "checksum": "15231916768216565527"}
}

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {cmd}")
        print(result.stderr.strip())
        return None
    return result.stdout.strip()

def parse_cpu_output(output):
    time_ms, living, checksum = None, None, None
    for line in output.splitlines():
        if "Execution time (ms):" in line:
            time_ms = float(line.split(":")[1].strip())
        elif "Living cells:" in line:
            living = int(line.split(":")[1].strip())
        elif "Checksum:" in line:
            checksum = line.split(":")[1].strip()
    return time_ms, living, checksum

def parse_cuda_output(output):
    gpu_name = "NVIDIA GPU"
    kernel_ms, total_ms, living, checksum = None, None, None, None
    for line in output.splitlines():
        if "GPU name:" in line:
            gpu_name = line.split(":")[1].strip()
        elif "Kernel execution time (ms):" in line:
            kernel_ms = float(line.split(":")[1].strip())
        elif "Total GPU time" in line:
            total_ms = float(line.split(":")[1].strip())
        elif "Execution time (ms):" in line and kernel_ms is None:
            kernel_ms = float(line.split(":")[1].strip())
        elif "Living cells:" in line:
            living = int(line.split(":")[1].strip())
        elif "Checksum:" in line:
            checksum = line.split(":")[1].strip()
    if total_ms is None and kernel_ms is not None:
        total_ms = kernel_ms
    return gpu_name, kernel_ms, total_ms, living, checksum

def main():
    print("=" * 78)
    print("  HIGH PERFORMANCE COMPUTING — BENCHMARK SUITE")
    print("  Conway's Game of Life: Sequential CPU vs CUDA GPU")
    print("=" * 78)

    # Detect compilers
    has_gpp = subprocess.run("which g++ || which clang++", shell=True, capture_output=True).returncode == 0
    has_nvcc = subprocess.run("which nvcc", shell=True, capture_output=True).returncode == 0

    if not has_gpp:
        print("[FATAL] C++ compiler (g++ or clang++) not found. Please install build essentials.")
        sys.exit(1)

    cxx = "g++" if subprocess.run("which g++", shell=True, capture_output=True).returncode == 0 else "clang++"
    print(f"[1/3] Compiling CPU implementation with {cxx} -O3 ...")
    res_cpu = run_command(f"{cxx} -O3 -std=c++17 cpu/game_of_life_cpu.cpp -o game_of_life_cpu")
    if res_cpu is None and not os.path.exists("game_of_life_cpu"):
        print("[FATAL] CPU compilation failed.")
        sys.exit(1)
    print("      ✓ CPU binary created: ./game_of_life_cpu")

    has_cuda = False
    if has_nvcc:
        print("[2/3] Compiling CUDA implementation with nvcc -O3 ...")
        res_cuda = run_command("nvcc -O3 cuda/game_of_life_cuda.cu -o game_of_life_cuda")
        if res_cuda is not None or os.path.exists("game_of_life_cuda"):
            has_cuda = True
            print("      ✓ CUDA binary created: ./game_of_life_cuda")
    else:
        print("[2/3] nvcc not detected on this machine.")
        print("      * CPU benchmarks will run locally.")
        print("      * For CUDA GPU benchmarks, run this script inside Google Colab.")

    print(f"\n[3/3] Executing benchmarks across grid sizes: {GRID_SIZES}")
    print(f"      Configurations: {DEFAULT_ITERATIONS} iterations, {DEFAULT_REPETITIONS} runs per size (averaging)")
    print("-" * 78)

    benchmark_records = []
    gpu_detected_name = "N/A (CPU Only)"

    for N in GRID_SIZES:
        cells = N * N
        print(f"\n>>> Benchmark Target: {N} x {N} ({cells:,} cells) | {DEFAULT_ITERATIONS} iterations")

        # --- Benchmark CPU ---
        cpu_times = []
        cpu_living, cpu_checksum = None, None
        for run_idx in range(DEFAULT_REPETITIONS):
            out = run_command(f"./game_of_life_cpu {N} {DEFAULT_ITERATIONS}")
            if out:
                t, l, c = parse_cpu_output(out)
                if t is not None:
                    cpu_times.append(t)
                    cpu_living, cpu_checksum = l, c
            time.sleep(0.05)

        avg_cpu_time = statistics.mean(cpu_times) if cpu_times else None
        print(f"    CPU Time (avg of {len(cpu_times)} runs): {avg_cpu_time:.3f} ms" if avg_cpu_time else "    CPU Time: FAILED")

        # --- Benchmark CUDA ---
        cuda_kernel_times = []
        cuda_total_times = []
        cuda_living, cuda_checksum = None, None
        if has_cuda:
            for run_idx in range(DEFAULT_REPETITIONS):
                out = run_command(f"./game_of_life_cuda {N} {DEFAULT_ITERATIONS}")
                if out:
                    g_name, k_t, tot_t, l, c = parse_cuda_output(out)
                    if g_name: gpu_detected_name = g_name
                    if k_t is not None: cuda_kernel_times.append(k_t)
                    if tot_t is not None: cuda_total_times.append(tot_t)
                    cuda_living, cuda_checksum = l, c
                time.sleep(0.05)

        avg_cuda_kernel = statistics.mean(cuda_kernel_times) if cuda_kernel_times else None
        avg_cuda_total = statistics.mean(cuda_total_times) if cuda_total_times else None

        if avg_cuda_kernel:
            print(f"    CUDA Kernel Time (avg): {avg_cuda_kernel:.3f} ms")
            print(f"    CUDA Total Time (avg):  {avg_cuda_total:.3f} ms")
            print(f"    GPU Hardware:          {gpu_detected_name}")

        # Correctness & Validation check
        ref = REFERENCE_CHECKSUMS.get(N, {})
        val_status = "PENDING"
        speedup_kernel = 0.0
        speedup_total = 0.0

        if avg_cpu_time:
            # Check against reference mathematical values
            cpu_matches_ref = (cpu_living == ref.get("living")) and (cpu_checksum == ref.get("checksum"))
            if has_cuda and avg_cuda_kernel:
                cuda_matches_cpu = (cpu_living == cuda_living) and (cpu_checksum == cuda_checksum)
                if cpu_matches_ref and cuda_matches_cpu:
                    val_status = "PASSED"
                else:
                    val_status = "MISMATCH"
                speedup_kernel = round(avg_cpu_time / avg_cuda_kernel, 2)
                speedup_total = round(avg_cpu_time / avg_cuda_total, 2)
                print(f"    [VALIDATION] {val_status} (CPU checksum == CUDA checksum)")
                print(f"    [SPEEDUP]    Kernel: {speedup_kernel}x | End-to-End: {speedup_total}x")
            else:
                val_status = "PASSED (CPU Ref)" if cpu_matches_ref else "MISMATCH"
                print(f"    [VALIDATION] {val_status} (Living: {cpu_living:,} | Checksum: {cpu_checksum})")

        record = {
            "gridSize": N,
            "cells": cells,
            "iterations": DEFAULT_ITERATIONS,
            "blockSize": "16x16",
            "cpuTimeMs": round(avg_cpu_time, 3) if avg_cpu_time else None,
            "gpuKernelTimeMs": round(avg_cuda_kernel, 3) if avg_cuda_kernel else None,
            "gpuTotalTimeMs": round(avg_cuda_total, 3) if avg_cuda_total else None,
            "speedup": speedup_kernel if avg_cuda_kernel else None,
            "speedupTotal": speedup_total if avg_cuda_total else None,
            "livingCells": cpu_living,
            "checksum": str(cpu_checksum),
            "validation": val_status,
            "gpuName": gpu_detected_name
        }
        benchmark_records.append(record)

    # Export to results/
    os.makedirs("results", exist_ok=True)

    # 1. Export JSON format
    json_path = "results/benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)
    print(f"\n[OUTPUT] Exported JSON to: {json_path}")

    # 2. Export CSV format
    csv_path = "results/benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write("grid_size,cells,iterations,block_size,cpu_time_ms,cuda_kernel_time_ms,cuda_total_time_ms,speedup_kernel,speedup_total,living_cells,checksum,validation,gpu_name\n")
        for r in benchmark_records:
            cpu_t = r["cpuTimeMs"] if r["cpuTimeMs"] is not None else ""
            k_t = r["gpuKernelTimeMs"] if r["gpuKernelTimeMs"] is not None else ""
            tot_t = r["gpuTotalTimeMs"] if r["gpuTotalTimeMs"] is not None else ""
            sp = r["speedup"] if r["speedup"] is not None else ""
            sp_tot = r["speedupTotal"] if r["speedupTotal"] is not None else ""
            f.write(f"{r['gridSize']},{r['cells']},{r['iterations']},{r['blockSize']},{cpu_t},{k_t},{tot_t},{sp},{sp_tot},{r['livingCells']},{r['checksum']},{r['validation']},{r['gpuName']}\n")
    print(f"[OUTPUT] Exported CSV to:  {csv_path}")

    # 3. Also update results/results.csv for backward compatibility
    legacy_csv_path = "results/results.csv"
    with open(legacy_csv_path, "w") as f:
        f.write("grid_size,iterations,block_size,cpu_time_ms,cuda_time_ms,speedup,living_cells,checksum,gpu_name\n")
        for r in benchmark_records:
            cpu_t = r["cpuTimeMs"] if r["cpuTimeMs"] is not None else ""
            k_t = r["gpuKernelTimeMs"] if r["gpuKernelTimeMs"] is not None else ""
            sp = r["speedup"] if r["speedup"] is not None else ""
            f.write(f"{r['gridSize']},{r['iterations']},{r['blockSize']},{cpu_t},{k_t},{sp},{r['livingCells']},{r['checksum']},{r['gpuName']}\n")
    print(f"[OUTPUT] Updated legacy:   {legacy_csv_path}")

    print("\n" + "=" * 78)
    print("  FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 78)
    header = f"{'Grid':^9} | {'Cells':^10} | {'CPU (ms)':^10} | {'CUDA Kernel':^11} | {'CUDA Total':^10} | {'Speedup':^9} | {'Validation':^10}"
    print(header)
    print("-" * 78)
    for r in benchmark_records:
        cpu_str = f"{r['cpuTimeMs']:.2f}" if r["cpuTimeMs"] is not None else "N/A"
        k_str = f"{r['gpuKernelTimeMs']:.2f}" if r["gpuKernelTimeMs"] is not None else "N/A"
        tot_str = f"{r['gpuTotalTimeMs']:.2f}" if r["gpuTotalTimeMs"] is not None else "N/A"
        sp_str = f"{r['speedup']}x" if r["speedup"] is not None else "N/A"
        print(f"{r['gridSize']:>4}x{r['gridSize']:<4} | {r['cells']:>10,} | {cpu_str:>10} | {k_str:>11} | {tot_str:>10} | {sp_str:>9} | {r['validation']:^10}")
    print("=" * 78)

if __name__ == "__main__":
    main()
