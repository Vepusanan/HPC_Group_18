#!/usr/bin/env python3
"""
Automated HPC Benchmark Runner for Conway's Game of Life
Runs CPU and CUDA implementations across all grid sizes, validates checksums,
calculates speedups, and updates results/results.csv.
"""

import subprocess
import re
import os
import sys

GRID_SIZES = [256, 512, 1024, 2048, 4096]
ITERATIONS = 100

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(result.stderr)
        return None
    return result.stdout

def parse_cpu_output(output):
    time_ms = None
    living = None
    checksum = None
    for line in output.splitlines():
        if "Execution time (ms):" in line:
            time_ms = float(line.split(":")[1].strip())
        elif "Living cells:" in line:
            living = int(line.split(":")[1].strip())
        elif "Checksum:" in line:
            checksum = line.split(":")[1].strip()
    return time_ms, living, checksum

def parse_cuda_output(output):
    gpu_name = None
    time_ms = None
    living = None
    checksum = None
    for line in output.splitlines():
        if "GPU name:" in line:
            gpu_name = line.split(":")[1].strip()
        elif "Execution time (ms):" in line:
            time_ms = float(line.split(":")[1].strip())
        elif "Living cells:" in line:
            living = int(line.split(":")[1].strip())
        elif "Checksum:" in line:
            checksum = line.split(":")[1].strip()
    return gpu_name, time_ms, living, checksum

def main():
    print("=" * 60)
    print("HPC Benchmark Suite: CPU vs CUDA")
    print("=" * 60)

    # Check compilers
    has_gpp = subprocess.run("which g++", shell=True, capture_output=True).returncode == 0
    has_nvcc = subprocess.run("which nvcc", shell=True, capture_output=True).returncode == 0

    if not has_gpp:
        print("Error: g++ not found.")
        sys.exit(1)

    print("Compiling CPU implementation (cpu/game_of_life_cpu.cpp)...")
    run_command("g++ -O3 cpu/game_of_life_cpu.cpp -o game_of_life_cpu")

    has_cuda_exe = False
    if has_nvcc:
        print("Compiling CUDA implementation (cuda/game_of_life_cuda.cu)...")
        res = run_command("nvcc -O3 cuda/game_of_life_cuda.cu -o game_of_life_cuda")
        if res is not None:
            has_cuda_exe = True
    else:
        print("Notice: nvcc not found. Skipping CUDA compilation (CPU-only mode).")

    results_lines = ["grid_size,iterations,block_size,cpu_time_ms,cuda_time_ms,speedup,living_cells,checksum,gpu_name\n"]

    for N in GRID_SIZES:
        print(f"\n--- Running Grid Size: {N} x {N} ({ITERATIONS} iterations) ---")
        
        # Run CPU
        print("  Running CPU...")
        cpu_out = run_command(f"./game_of_life_cpu {N} {ITERATIONS}")
        cpu_time, cpu_living, cpu_checksum = parse_cpu_output(cpu_out) if cpu_out else (None, None, None)
        print(f"    CPU Time: {cpu_time:.3f} ms | Living: {cpu_living} | Checksum: {cpu_checksum}")

        # Run CUDA
        cuda_time, cuda_living, cuda_checksum, gpu_name = None, None, None, ""
        if has_cuda_exe:
            print("  Running CUDA...")
            cuda_out = run_command(f"./game_of_life_cuda {N} {ITERATIONS}")
            if cuda_out:
                gpu_name, cuda_time, cuda_living, cuda_checksum = parse_cuda_output(cuda_out)
                print(f"    CUDA Time: {cuda_time:.3f} ms | Living: {cuda_living} | Checksum: {cuda_checksum}")
                print(f"    GPU: {gpu_name}")

                # Correctness validation check
                if cpu_living == cuda_living and cpu_checksum == cuda_checksum:
                    speedup = round(cpu_time / cuda_time, 2) if cuda_time and cuda_time > 0 else "N/A"
                    print(f"    Validation: PASSED (Match 100%) | Speedup: {speedup}x")
                else:
                    speedup = "MISMATCH"
                    print("    Validation: FAILED (Living count or checksum diverged!)")
            else:
                speedup = ""
        else:
            speedup = ""

        speedup_str = str(speedup) if speedup is not None else ""
        cuda_time_str = f"{cuda_time:.3f}" if cuda_time is not None else ""
        cpu_time_str = f"{cpu_time:.3f}" if cpu_time is not None else ""
        living_str = str(cpu_living) if cpu_living is not None else ""
        checksum_str = str(cpu_checksum) if cpu_checksum is not None else ""

        results_lines.append(f"{N},{ITERATIONS},16x16,{cpu_time_str},{cuda_time_str},{speedup_str},{living_str},{checksum_str},{gpu_name}\n")

    # Write to results/results.csv
    os.makedirs("results", exist_ok=True)
    with open("results/results.csv", "w") as f:
        f.writelines(results_lines)

    print("\n" + "=" * 60)
    print("Benchmark complete! Results saved to results/results.csv")
    print("=" * 60)
    with open("results/results.csv") as f:
        print(f.read())

if __name__ == "__main__":
    main()
