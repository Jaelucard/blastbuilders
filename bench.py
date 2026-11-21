#!/usr/bin/env python3
"""Local benchmark harness matching the scorer's exact Docker flags + CPU accounting."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="sample", help="Directory with tests.json & commits/")
    parser.add_argument("--answers", default=None, help="Path to answers.json (default: <data-dir>/answers.json)")
    parser.add_argument("--no-build", action="store_true", help="Skip docker build")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (reports median)")
    parser.add_argument("--image", default="blast-radius-entry")
    args = parser.parse_args()

    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    answers_path = Path(args.answers).resolve() if args.answers else (data_dir / "answers.json")

    if not args.no_build:
        t_b0 = time.perf_counter()
        subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "--tag", args.image, "."],
            check=True,
        )
        print(f"[bench] build took {time.perf_counter() - t_b0:.2f}s", file=sys.stderr)

    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
        "--cpus", "2", "--memory", "4096m", "--memory-swap", "4096m", "--pids-limit", "256",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534", "--env", "HOME=/tmp",
        "--volume", f"{data_dir / 'commits'}:/data/commits:ro",
        "--volume", f"{data_dir / 'tests.json'}:/data/tests.json:ro",
        args.image,
    ]

    totals = []
    for r in range(args.runs):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wall = time.perf_counter() - t0
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            print(f"[bench] run {r+1} FAILED exit={proc.returncode} wall={wall:.3f}s", file=sys.stderr)
            print(stderr_text, file=sys.stderr)
            return proc.returncode

        out_path = root / "output.jsonl"
        out_path.write_bytes(proc.stdout)

        cpu_match = re.search(r"\[cpu\]\s+total_cpu_s=([0-9.]+)", stderr_text)
        cpu_s = float(cpu_match.group(1)) if cpu_match else None

        print(f"\n--- Run {r+1}/{args.runs} (wall={wall:.3f}s, stdout={len(proc.stdout)} bytes) ---")
        if stderr_text.strip():
            for line in stderr_text.strip().splitlines():
                if line.startswith("[sel]") or line.startswith("[cpu]"):
                    print(f"  {line}")

        score_cmd = [sys.executable, "score.py", str(data_dir), str(answers_path), str(out_path)]
        if cpu_s is not None:
            score_cmd += ["--cpu-seconds", f"{cpu_s:.6f}"]
        res = subprocess.run(score_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res.stdout.strip())
        if res.stderr.strip():
            print(res.stderr.strip(), file=sys.stderr)
        if res.returncode != 0:
            return res.returncode

        m_tot = re.search(r"Total:\s+([0-9.]+)\s+s", res.stdout)
        m_sim = re.search(r"Simulated:\s+([0-9.]+)\s+s", res.stdout)
        if m_tot:
            totals.append(float(m_tot.group(1)))
        elif m_sim:
            totals.append(float(m_sim.group(1)))

    if len(totals) > 1:
        totals_sorted = sorted(totals)
        med = totals_sorted[len(totals_sorted) // 2]
        print(f"\n=== Median of {len(totals)} runs: {med:.3f} s (all: {totals}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
