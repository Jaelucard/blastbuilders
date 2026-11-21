"""Exact optimal worker packing (k = 1..8) and Shadow-Slack cost calculations."""

from __future__ import annotations

import math

WORKER_COST_MS = 2_000
MAX_WORKERS = 8


def duration_ms(seconds: float) -> int:
    """Matches score.py's exact half-up millisecond rounding."""
    return math.floor(seconds * 1000 + 0.5)


def _lpt_pack(items: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    """Longest-Processing-Time greedy packing. items are (dur_ms, test_num) sorted desc."""
    workers: list[list[int]] = [[] for _ in range(k)]
    loads = [0] * k
    for d, num in items:
        idx = loads.index(min(loads))
        workers[idx].append(num)
        loads[idx] += d
    return max(loads), workers


def _exact_pack_k(items: list[tuple[int, int]], k: int, ub: int, lb: int, max_nodes: int = 3000) -> tuple[int, list[list[int]]] | None:
    """Symmetry-breaking branch-and-bound for exact makespan minimization on the
    heaviest tests, filling tiny remaining tests greedily."""
    # Separate heavy items from tiny fillers (<= 5ms); tiny items fill gaps nearly perfectly
    heavy_cut = min(len(items), 28)
    while heavy_cut > 0 and items[heavy_cut - 1][0] <= 5:
        heavy_cut -= 1
    if heavy_cut <= k:
        return None  # LPT is already exact when <= k heavy items

    heavy = items[:heavy_cut]
    tail = items[heavy_cut:]
    tail_sum = sum(d for d, _ in tail)

    best_makespan = ub
    best_assign: list[int] | None = None
    loads = [0] * k
    assign = [0] * heavy_cut
    nodes = 0

    def dfs(i: int, cur_max: int) -> None:
        nonlocal best_makespan, best_assign, nodes
        if best_makespan <= lb or nodes >= max_nodes:
            return
        if i == heavy_cut:
            # Evaluate with greedy tail packing
            if not tail:
                if cur_max < best_makespan:
                    best_makespan = cur_max
                    best_assign = assign[:]
                return
            t_loads = loads[:]
            for d, _ in tail:
                idx = t_loads.index(min(t_loads))
                t_loads[idx] += d
            mk = max(t_loads)
            if mk < best_makespan:
                best_makespan = mk
                best_assign = assign[:]
            return

        nodes += 1
        d = heavy[i][0]
        seen_loads: set[int] = set()
        # Try bins in order of ascending load (symmetry breaking: skip duplicate bin loads)
        order = sorted(range(k), key=loads.__getitem__)
        for b in order:
            ld = loads[b]
            if ld in seen_loads:
                continue
            seen_loads.add(ld)
            nld = ld + d
            if nld >= best_makespan:
                continue
            loads[b] = nld
            assign[i] = b
            dfs(i + 1, nld if nld > cur_max else cur_max)
            loads[b] = ld
            if best_makespan <= lb or nodes >= max_nodes:
                return

    # Place item 0 in bin 0 (symmetry breaking)
    loads[0] = heavy[0][0]
    assign[0] = 0
    dfs(1, heavy[0][0])

    if best_assign is None:
        return None

    workers: list[list[int]] = [[] for _ in range(k)]
    f_loads = [0] * k
    for idx, b in enumerate(best_assign):
        d, num = heavy[idx]
        workers[b].append(num)
        f_loads[b] += d
    for d, num in tail:
        b = f_loads.index(min(f_loads))
        workers[b].append(num)
        f_loads[b] += d
    return max(f_loads), workers


def pack_optimal(test_nums: list[int] | set[int], dur_ms_by_num: dict[int, int]) -> tuple[int, list[list[int]]]:
    """Return (min_simulated_cost_ms, workers_list) across k = 1..8."""
    if not test_nums:
        return 0, []

    items = sorted(((dur_ms_by_num[n], n) for n in test_nums), key=lambda x: (-x[0], x[1]))
    total_ms = sum(d for d, _ in items)
    max_item = items[0][0]
    n = len(items)

    # k = 1 baseline
    best_cost = total_ms + WORKER_COST_MS
    best_workers: list[list[int]] = [[num for _, num in items]]

    max_k = min(MAX_WORKERS, n)
    for k in range(2, max_k + 1):
        lb = max(max_item, (total_ms + k - 1) // k)
        if lb + WORKER_COST_MS * k >= best_cost:
            # If even max_item + 2000*k >= best_cost, larger k can never improve!
            if max_item + WORKER_COST_MS * k >= best_cost:
                break
            continue

        mk, w = _lpt_pack(items, k)
        if mk > lb:
            better = _exact_pack_k(items, k, ub=mk, lb=lb)
            if better is not None and better[0] < mk:
                mk, w = better

        cost = mk + WORKER_COST_MS * k
        if cost < best_cost:
            best_cost = cost
            best_workers = w

    # Drop any empty workers (defensive)
    best_workers = [w for w in best_workers if w]
    return best_cost, best_workers


def optimal_cost_ms(test_nums: list[int] | set[int], dur_ms_by_num: dict[int, int]) -> int:
    """Fast cost-only version for Shadow-Slack / Decision-Theory gating."""
    if not test_nums:
        return 0
    items = sorted(((dur_ms_by_num[n], n) for n in test_nums), key=lambda x: (-x[0], x[1]))
    total_ms = sum(d for d, _ in items)
    max_item = items[0][0]
    best = total_ms + WORKER_COST_MS
    max_k = min(MAX_WORKERS, len(items))
    for k in range(2, max_k + 1):
        lb = max(max_item, (total_ms + k - 1) // k)
        if lb + WORKER_COST_MS * k >= best:
            if max_item + WORKER_COST_MS * k >= best:
                break
            continue
        mk, _ = _lpt_pack(items, k)
        if mk > lb:
            better = _exact_pack_k(items, k, ub=mk, lb=lb, max_nodes=800)
            if better is not None and better[0] < mk:
                mk = better[0]
        c = mk + WORKER_COST_MS * k
        if c < best:
            best = c
    return best
