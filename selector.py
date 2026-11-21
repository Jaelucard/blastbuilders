#!/usr/bin/env python3
"""Blast Radius selector entrypoint (V3).

Tiered zero-waste execution order:
  - Tier 0: Provably-inert commits -> emit [] (0 CPU, 0 simulated).
  - Pre-import immutable heavy C libs (duckdb, numpy, pandas) ONCE if needed.
  - Tier 1: Commits touching core `sqlglot` startup modules (fork BEFORE parent imports sqlglot).
  - Import `sqlglot` ONCE in parent.
  - Tier 2: Commits touching lazy sqlglot/tests modules (CoW fork with 0ms sqlglot import!).
  - Tier 3: Pure optimizer `.sql` fixture commits (zero-fork in-process verifier, ~3ms/commit!).
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ["SKIP_INTEGRATION"] = "0"

APP_DIR = Path("/app")
SQLGLOT_ROOT = APP_DIR / "sqlglot"
MAPS_PATH = APP_DIR / "maps" / "maps.zlib"
DATA_DIR = Path("/data")


def _read_total_cpu_seconds() -> float:
    cgroup_stat = Path("/sys/fs/cgroup/cpu.stat")
    if cgroup_stat.is_file():
        try:
            for line in cgroup_stat.read_text().splitlines():
                if line.startswith("usage_usec "):
                    return int(line.split()[1]) / 1_000_000.0
        except Exception:
            pass
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    r2 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r1.ru_utime + r1.ru_stime + r2.ru_utime + r2.ru_stime


def _emit_line(commit_id: str, workers: list[list[int]], valid_nums: set[int]) -> None:
    clean_workers: list[list[int]] = []
    seen: set[int] = set()
    for w in workers:
        cw = []
        for n in w:
            if isinstance(n, int) and not isinstance(n, bool) and n in valid_nums and n not in seen:
                seen.add(n)
                cw.append(n)
        if cw:
            clean_workers.append(cw)
    if len(clean_workers) > 8:
        merged = clean_workers[:8]
        for idx, extra in enumerate(clean_workers[8:]):
            merged[idx % 8].extend(extra)
        clean_workers = merged
    line = json.dumps({"commit": commit_id, "workers": clean_workers}, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main() -> None:
    t_wall0 = time.monotonic()

    tests_data = json.loads((DATA_DIR / "tests.json").read_text("utf-8"))
    tests = tests_data["tests"]

    from impact import ImpactMaps, analyze_commit
    from packer import duration_ms, pack_optimal
    from verifier import PureFixtureRunner, verify_commit_in_fork

    dur_ms_by_num = {t["number"]: duration_ms(t["seconds"]) for t in tests}
    valid_nums = set(dur_ms_by_num.keys())
    maps = ImpactMaps(MAPS_PATH, tests)

    patch_paths = sorted((DATA_DIR / "commits").glob("*.patch"))
    n_commits = len(patch_paths)
    wall_budget = 135.0 if n_commits <= 45 else 210.0
    hard_deadline = t_wall0 + wall_budget

    analyses = []
    for pp in patch_paths:
        cid = pp.stem
        patch_text = pp.read_text("utf-8", errors="replace")
        analyses.append(analyze_commit(cid, patch_text, SQLGLOT_ROOT, maps))

    fix_runner = PureFixtureRunner()

    inert_list = []
    tier1_clean = []
    tier2_warm = []
    tier3_pure_fix = []
    fallback_list = []

    needs_heavy_libs = False
    for a in analyses:
        if a.is_inert or not a.candidate_nums:
            inert_list.append(a)
        elif a.apply_failed:
            fallback_list.append(a)
        elif fix_runner.can_handle(a, maps):
            tier3_pure_fix.append(a)
            needs_heavy_libs = True
        else:
            if not needs_heavy_libs:
                for n in a.candidate_nums:
                    tf = maps.num_to_test[n]["file"]
                    if tf in ("tests/test_optimizer.py", "tests/test_executor.py"):
                        needs_heavy_libs = True
                        break
            if a.needs_clean_sqlglot:
                tier1_clean.append(a)
            else:
                tier2_warm.append(a)

    for a in inert_list:
        _emit_line(a.commit_id, [], valid_nums)
        print(f"[sel] {a.commit_id}: INERT -> 0 tests", file=sys.stderr)

    for a in fallback_list:
        cost_ms, workers = pack_optimal(a.candidate_nums, dur_ms_by_num)
        _emit_line(a.commit_id, workers, valid_nums)
        print(f"[sel] {a.commit_id}: APPLY_FAILED -> {len(a.candidate_nums)} tests ({cost_ms/1000:.3f}s)", file=sys.stderr)

    if needs_heavy_libs and (tier1_clean or tier2_warm or tier3_pure_fix):
        import duckdb  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401

    def _run_forked(a):
        c0 = _read_total_cpu_seconds()
        rem_wall = hard_deadline - time.monotonic()
        if rem_wall < 5.0:
            selected = list(a.candidate_nums)
            stats = {"fallback": "wall_budget"}
        else:
            per_commit_lim = min(35.0, max(5.0, rem_wall * 0.4))
            selected, stats = verify_commit_in_fork(a, maps, dur_ms_by_num, commit_timeout_s=per_commit_lim)
        cost_ms, workers = pack_optimal(selected, dur_ms_by_num)
        _emit_line(a.commit_id, workers, valid_nums)
        dcpu = _read_total_cpu_seconds() - c0
        print(
            f"[sel] {a.commit_id}: cand={len(a.candidate_nums)} -> sel={len(selected)} "
            f"sim={cost_ms/1000:.3f}s cpu={dcpu:.3f}s w={len(workers)} stats={stats}",
            file=sys.stderr,
        )

    # Tier 1: core sqlglot edits (before parent imports sqlglot)
    for a in tier1_clean:
        _run_forked(a)

    # Warm-import sqlglot ONCE in parent
    if tier2_warm or tier3_pure_fix:
        if str(SQLGLOT_ROOT) not in sys.path:
            sys.path.insert(0, str(SQLGLOT_ROOT))
        import sqlglot  # noqa: F401

        # Tier 2: forked verification with 0ms sqlglot import
        for a in tier2_warm:
            _run_forked(a)

        # Tier 3: zero-fork pure optimizer fixture verification!
        for a in tier3_pure_fix:
            c0 = _read_total_cpu_seconds()
            selected, stats = fix_runner.verify(a, maps)
            cost_ms, workers = pack_optimal(selected, dur_ms_by_num)
            _emit_line(a.commit_id, workers, valid_nums)
            dcpu = _read_total_cpu_seconds() - c0
            print(
                f"[sel] {a.commit_id}: cand={len(a.candidate_nums)} -> sel={len(selected)} "
                f"sim={cost_ms/1000:.3f}s cpu={dcpu:.3f}s w={len(workers)} stats={stats}",
                file=sys.stderr,
            )

    cpu_s = _read_total_cpu_seconds()
    wall_s = time.monotonic() - t_wall0
    print(f"[cpu] total_cpu_s={cpu_s:.6f} wall_s={wall_s:.3f} commits={n_commits}", file=sys.stderr)


if __name__ == "__main__":
    main()
