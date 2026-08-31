#!/usr/bin/env python3
import psutil
import subprocess
import time
import csv
import sys
from datetime import datetime


def get_all_children(pid):
    """Recursively get all child PIDs of a given PID."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        return [parent] + children
    except psutil.NoSuchProcess:
        return []


def get_gpu_usage(pids):
    """Get total GPU utilization (all GPUs) and GPU memory used by the given PIDs."""
    # Get GPU memory used by the monitored processes
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        gpu_info = result.stdout.strip().split("\n")
        total_gpu_mem = 0
        for line in gpu_info:
            if not line.strip():
                continue
            pid_str, mem_str = [x.strip() for x in line.split(",")]
            if int(pid_str) in pids:
                total_gpu_mem += float(mem_str)
    except Exception:
        total_gpu_mem = 0.0

    return total_gpu_mem


def monitor(pid, csv_path, interval=1):
    start_time = time.time()
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["timestamp", "elapsed_sec", "cpu_percent", "mem_mb", "gpu_mem_mb"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = time.time() - start_time
            procs = get_all_children(pid)
            if not procs:
                print("Process ended.")
                break
            cpu = 0.0
            mem = 0.0
            pids = []
            for p in procs:
                try:
                    cpu += p.cpu_percent(interval=0.1)
                    mem += p.memory_info().rss / (1024 * 1024)
                    pids.append(p.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                    continue
            gpu_mem = get_gpu_usage(pids)
            writer.writerow(
                {
                    "timestamp": now,
                    "elapsed_sec": f"{elapsed:.1f}",
                    "cpu_percent": f"{cpu:.1f}",
                    "mem_mb": f"{mem:.1f}",
                    "gpu_mem_mb": f"{gpu_mem:.1f}",
                }
            )
            csvfile.flush()
            time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <PID> <output.csv>")
        sys.exit(1)
    pid = int(sys.argv[1])
    csv_path = sys.argv[2]
    monitor(pid, csv_path)
