"""Stage 2: Fast isolated in-process test verifier (V3).

Includes:
  - CoW os.fork() isolation + symlink-shadow overlay
  - Zero-fork in-process verifier for pure fixture-only commits
  - Shared tpcds_opt_cache between test_tpcds and test_execute_tpcds
  - Fast (columns, nrows) check before opening DuckDB
  - Per-section test_scope fast verifier (4 sections -> ~2ms instead of 2533ms!)
  - Calibrated est_cpu + Shadow-Slack + Decision-Theory gating
  - Sound Python type/Enum proof for pure `is`<->`==` / `is not`<->`!=` edits
"""

from __future__ import annotations

import ast
import concurrent.futures as cf
import contextlib
import enum
import gc
import hashlib
import importlib
import os
import pickle
import select
import signal
import sys
import time
import tracemalloc
import unittest
from pathlib import Path

from impact import CommitAnalysis, ImpactMaps
from packer import optimal_cost_ms
from patch_utils import (
    OverlayWorkspace,
    parse_sql_fixture_pairs_text,
    parse_sql_fixtures_text,
)

SQLGLOT_ROOT = Path("/app/sqlglot").resolve()
TABLES_PKL = Path("/app/maps/tables.pkl")

BENCHMARK_TESTS_FOR_EQUIV = frozenset({
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_optimizer.py::TestOptimizer::test_tpcds",
    "tests/test_optimizer.py::TestOptimizer::test_scope",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_columns",
    "tests/test_parser.py::TestParser::test_max_nodes",
    "tests/test_generator.py::TestGenerator::test_generator_memory",
})


class _FastFail(BaseException):
    pass


class _TestTimeout(BaseException):
    pass


def _alarm_handler(signum, frame):
    raise _TestTimeout()


class _SyncFuture:
    __slots__ = ("_val", "_exc")

    def __init__(self, val=None, exc=None):
        self._val = val
        self._exc = exc

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._val

    def exception(self, timeout=None):
        return self._exc

    def done(self):
        return True

    def add_done_callback(self, fn):
        fn(self)


class FastSyncPool:
    def __init__(self, *args, initializer=None, initargs=(), do_pickle=False, **kwargs):
        self._do_pickle = do_pickle
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def shutdown(self, wait=True, cancel_futures=False):
        pass

    def submit(self, fn, *args, **kwargs):
        try:
            if self._do_pickle:
                args = pickle.loads(pickle.dumps(args))
                if kwargs:
                    kwargs = pickle.loads(pickle.dumps(kwargs))
            res = fn(*args, **kwargs)
            if self._do_pickle:
                res = pickle.loads(pickle.dumps(res))
            return _SyncFuture(val=res)
        except Exception as e:
            return _SyncFuture(exc=e)


def _hash_str(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()


def _normalize_table_hash(table) -> bytes:
    rows = []
    for r in table.rows:
        rows.append(tuple(
            ("F", round(c, 8)) if isinstance(c, float) else
            ("N", None) if c is None else
            (type(c).__name__, str(c))
            for c in r
        ))
    return hashlib.blake2b(
        pickle.dumps((tuple(table.columns), rows)), digest_size=16
    ).digest()


def _rewrite_sys_module_paths(ov_root: Path) -> None:
    base_str = str(SQLGLOT_ROOT)
    base_pref = base_str + os.sep
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        p_attr = getattr(mod, "__path__", None)
        if p_attr is None:
            continue
        try:
            paths = list(p_attr)
        except TypeError:
            continue
        changed = False
        new_paths = []
        for p in paths:
            if p == base_str:
                new_paths.append(str(ov_root))
                changed = True
            elif p.startswith(base_pref):
                rel = p[len(base_pref):]
                new_paths.append(str(ov_root / rel))
                changed = True
            else:
                new_paths.append(p)
        if changed:
            mod.__path__ = new_paths


def _impacted_bits_from_subcov(
    sub_cov: dict[str, dict[int, int]],
    py_changed_lines: dict[str, set[int]],
) -> int:
    mask = 0
    for rel, lines in py_changed_lines.items():
        fl = sub_cov.get(rel)
        if fl:
            for ln in lines:
                mask |= fl.get(ln, 0)
    return mask


def _bits_to_list(mask: int) -> list[int]:
    out = []
    m = mask
    while m:
        lsb = m & -m
        out.append(lsb.bit_length() - 1)
        m ^= lsb
    return out


OPT_TEST_TO_FIXTURES: dict[str, list[str]] = {
    "test_optimize": ["optimizer/optimizer.sql"],
    "test_qualify_columns": ["optimizer/qualify_columns.sql", "optimizer/qualify_columns_ddl.sql"],
    "test_merge_subqueries": ["optimizer/merge_subqueries.sql"],
    "test_canonicalize": ["optimizer/canonicalize.sql"],
    "test_canonicalize_internal_names": ["optimizer/canonicalize_internal_names.sql"],
    "test_simplify": ["optimizer/simplify.sql"],
    "test_unnest_subqueries": ["optimizer/unnest_subqueries.sql"],
    "test_tpcds": ["optimizer/tpc-ds/tpc-ds.sql"],
    "test_tpch": ["optimizer/tpc-h/tpc-h.sql"],
    "test_eliminate_joins": ["optimizer/eliminate_joins.sql"],
    "test_pushdown_projection": ["optimizer/pushdown_projections.sql"],
    "test_pushdown_predicates": ["optimizer/pushdown_predicates.sql"],
    "test_annotate_types": ["optimizer/annotate_types.sql"],
    "test_annotate_funcs": ["optimizer/annotate_functions.sql"],
    "test_isolate_table_selects": ["optimizer/isolate_table_selects.sql"],
    "test_qualify_tables": ["optimizer/qualify_tables.sql"],
    "test_normalize": ["optimizer/normalize.sql"],
    "test_qualify_columns__with_invisible": ["optimizer/qualify_columns__with_invisible.sql"],
    "test_pushdown_cte_alias_columns": ["optimizer/pushdown_cte_alias_columns.sql"],
    "test_qualify_columns__invalid": ["optimizer/qualify_columns__invalid.sql"],
    "test_normalize_identifiers": ["optimizer/normalize_identifiers.sql"],
    "test_quote_identifiers": ["optimizer/quote_identifiers.sql"],
    "test_optimize_joins": ["optimizer/optimize_joins.sql"],
    "test_eliminate_ctes": ["optimizer/eliminate_ctes.sql"],
    "test_eliminate_subqueries": ["optimizer/eliminate_subqueries.sql"],
}


def _compute_fixture_filter(
    fixture_rel: str,
    analysis: CommitAnalysis,
    maps: ImpactMaps,
) -> set[int] | None:
    if not analysis.touches_any_py:
        ch = analysis.fixture_changed_indices.get(fixture_rel)
        if ch is not None:
            return set(ch)
        return set()

    extra = set(analysis.fixture_changed_indices.get(fixture_rel, ()))
    non_dialect_top = [
        f for f in analysis.py_class_or_top_files
        if not (
            f.startswith("sqlglot/dialects/")
            or f.startswith("sqlglot/parsers/")
            or f.startswith("sqlglot/generators/")
            or f.startswith("sqlglot/typing/")
        )
        or f == "sqlglot/dialects/dialect.py"
    ]
    if non_dialect_top:
        return None

    if fixture_rel == "optimizer/tpc-ds/tpc-ds.sql":
        if analysis.py_class_or_top_files:
            return None
        mask = _impacted_bits_from_subcov(maps.tpcds_all_opt_cov, analysis.py_changed_lines)
        return set(_bits_to_list(mask)) | extra

    finfo = maps.opt_fixtures.get(fixture_rel)
    if finfo is None:
        return None

    mask = _impacted_bits_from_subcov(finfo["cov"], analysis.py_changed_lines)
    if analysis.py_dialect_mods:
        dials = finfo["dialects"]
        for dm in analysis.py_dialect_mods:
            mask |= dials.get(dm, 0)
    for f in analysis.py_class_or_top_files:
        if f.startswith("sqlglot/typing/"):
            d_stem = Path(f).stem
            mask |= finfo["dialects"].get(f"sqlglot.dialects.{d_stem}", 0)
            if d_stem in ("__init__", "dialect"):
                return None

    return set(_bits_to_list(mask)) | extra


def _estimate_verify_cpu_ms(
    num: int,
    dur_ms: int,
    analysis: CommitAnalysis,
    maps: ImpactMaps,
) -> int:
    tid = maps.num_to_test[num]["id"]
    tf, _, meth = tid.split("::")

    if meth == "test_execute_tpch":
        return 250 if "optimizer/tpc-h/tpc-h.sql" in analysis.fixture_changed_indices else 30

    if meth == "test_optimized_tpch":
        return 200 if (analysis.touches_duckdb or "optimizer/tpc-h/tpc-h.sql" in analysis.fixture_changed_indices) else 20

    if meth == "test_execute_tpcds":
        if analysis.touches_serde:
            return 30
        if not analysis.touches_executor_or_planner and "optimizer/tpc-ds/tpc-ds.sql" not in analysis.fixture_changed_indices:
            if not analysis.py_class_or_top_files:
                mask = _impacted_bits_from_subcov(maps.tpcds_opt_cov, analysis.py_changed_lines)
                return max(30, min(1900, mask.bit_count() * 32))
            return 1800
        if not analysis.py_class_or_top_files:
            mask = _impacted_bits_from_subcov(maps.tpcds_exec_cov, analysis.py_changed_lines)
            qs = _bits_to_list(mask)
            return max(80, sum(maps.tpcds_exec_queries[i]["dur_ms"] for i in qs))
        return 11500

    if meth == "test_generator_memory":
        if not analysis.py_class_or_top_files:
            mask = _impacted_bits_from_subcov(maps.gen_mem_cov, analysis.py_changed_lines)
            return max(50, mask.bit_count() * 190)
        if analysis.py_dialect_mods and "sqlglot.dialects.dialect" not in analysis.py_dialect_mods:
            return max(190, len(analysis.py_dialect_mods) * 190)
        return 6500

    if tf == "tests/test_optimizer.py" and meth == "test_scope":
        if not analysis.py_class_or_top_files and maps.scope_sec_cov:
            smask = _impacted_bits_from_subcov(maps.scope_sec_cov, analysis.py_changed_lines)
            # Sections 0 and 3 are ~2ms each; section 1 is ~1500ms; section 2 is ~950ms
            cost = 0
            if smask & 1:
                cost += 3
            if smask & 2:
                cost += 1500
            if smask & 4:
                cost += 950
            if smask & 8:
                cost += 5
            return max(3, cost)
        return 2400

    if tid == "tests/test_parser.py::TestParser::test_max_nodes":
        return 28000

    if tf == "tests/test_optimizer.py" and meth in OPT_TEST_TO_FIXTURES:
        if not analysis.touches_any_py:
            return 10
        frels = OPT_TEST_TO_FIXTURES[meth]
        total_est = 0
        total_count = 0
        for fr in frels:
            flt = _compute_fixture_filter(fr, analysis, maps)
            if flt is None:
                return max(25, int(dur_ms * 0.65))
            if fr == "optimizer/tpc-ds/tpc-ds.sql":
                cnt = 99
            else:
                cnt = max(1, maps.opt_fixtures.get(fr, {}).get("count", 100))
            total_est += len(flt)
            total_count += cnt
        # Proportional estimate from recorded duration (e.g. annotate_funcs: 1963ms / 1973 pairs ~ 1ms/pair)
        est = max(5, int(dur_ms * (total_est / max(1, total_count)) * 0.8))
        return min(dur_ms, est)

    return max(2, int(dur_ms * 0.75))


def _verify_identity_equivalence_sound(analysis: CommitAnalysis, maps: ImpactMaps) -> bool:
    """Runtime-verified soundness proof for commits whose ENTIRE AST diff is
    swapping `is`<->`==` or `is not`<->`!=` on a typed attribute or Enum member.
    Returns True ONLY if Python's exact runtime class definitions prove equivalence."""
    if not analysis.all_edits_are_identity_op or not analysis.pure_identity_edits:
        return False
    for ed in analysis.pure_identity_edits:
        mod_name = maps.rel_to_mod.get(ed.rel_path)
        if not mod_name:
            return False
        try:
            mod = importlib.import_module(mod_name)
        except BaseException:
            return False

        if ed.kind == "none" and ed.type_hint_name:
            # Resolve the type annotation class (check module globals & TYPE_CHECKING imports!)
            t_cls = getattr(mod, ed.type_hint_name, None)
            if t_cls is None and ed.type_hint_name == "Table":
                try:
                    t_cls = getattr(importlib.import_module("sqlglot.executor.table"), "Table", None)
                except BaseException:
                    t_cls = None
            if not isinstance(t_cls, type):
                return False
            # Soundness check: T.__eq__ MUST be object.__eq__ AND no subclass overrides __eq__!
            if t_cls.__eq__ is not object.__eq__:
                return False
            stack = list(t_cls.__subclasses__())
            while stack:
                sub = stack.pop()
                if sub.__eq__ is not object.__eq__:
                    return False
                stack.extend(sub.__subclasses__())
            continue

        if ed.kind == "enum" and ed.enum_cls_name and ed.all_class_assigns_safe:
            e_cls = getattr(mod, ed.enum_cls_name, None)
            if not (isinstance(e_cls, type) and issubclass(e_cls, enum.Enum)):
                return False
            # Must have unique member values (no aliases)
            vals = [m.value for m in e_cls]
            if len(set(vals)) != len(vals):
                return False
            # Verify enclosing class default attribute (if any) is also an instance of e_cls
            if ed.enclosing_class and ed.self_attr:
                owner_cls = getattr(mod, ed.enclosing_class, None)
                if owner_cls is None:
                    return False
                upper_attr = ed.self_attr.upper()
                if hasattr(owner_cls, upper_attr):
                    default_v = getattr(owner_cls, upper_attr)
                    if not isinstance(default_v, e_cls):
                        return False
            continue

        return False
    return True


def _run_child_verification(
    analysis: CommitAnalysis,
    maps: ImpactMaps,
    dur_ms_by_num: dict[int, int],
    ov_root: Path,
    commit_deadline: float,
) -> tuple[list[int], dict]:
    sys.path.insert(0, str(ov_root))
    os.environ["PYTHONPATH"] = str(ov_root)
    os.environ["SKIP_INTEGRATION"] = "0"
    _rewrite_sys_module_paths(ov_root)

    do_pickle = analysis.touches_serde

    class _BoundPool(FastSyncPool):
        def __init__(self, *a, **kw):
            super().__init__(*a, do_pickle=do_pickle, **kw)

    def _fake_wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
        return (set(fs), set())

    def _fake_as_completed(fs, timeout=None):
        return iter(list(fs))

    cf.ProcessPoolExecutor = _BoundPool
    cf.wait = _fake_wait
    cf.as_completed = _fake_as_completed

    @contextlib.contextmanager
    def _fast_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        try:
            yield
        except unittest.SkipTest:
            pass
        except Exception as e:
            raise _FastFail(e) from e

    unittest.TestCase.subTest = _fast_subtest
    signal.signal(signal.SIGALRM, _alarm_handler)

    confirmed_fail: set[int] = set(analysis.confirmed_fail_nums)
    remaining: set[int] = set(analysis.candidate_nums) - confirmed_fail

    # Step 1: Import smoke-check for all patched non-test .py modules
    for rel, content in analysis.patched_files.items():
        if not rel.endswith(".py") or rel.startswith("tests/") or rel.startswith("sqlglotc/"):
            continue
        if content is None or rel in analysis.syntax_error_files:
            continue
        mod_name = maps.rel_to_mod.get(rel)
        if not mod_name:
            continue
        try:
            signal.alarm(5)
            importlib.import_module(mod_name)
            signal.alarm(0)
        except BaseException:
            signal.alarm(0)
            eager_nums = maps.mask_to_nums(maps.rev_eager_mask.get(rel, 0))
            confirmed_fail.update(eager_nums)
            remaining -= confirmed_fail

    # Sound runtime proof for pure identity-op edits (`is`<->`==` / `is not`<->`!=`)
    identity_equiv_proven = False
    if analysis.all_edits_are_identity_op and not confirmed_fail:
        identity_equiv_proven = _verify_identity_equivalence_sound(analysis, maps)

    # Shared caches inside this commit's forked child
    loaded_modules: dict[str, object | Exception] = {}
    setup_classes_done: dict[tuple[str, str], object | Exception] = {}
    preloaded_tables_cache: dict | None = None
    duckdb_conns: dict[str, object] = {}
    # Shared TPC-DS optimize cache between test_tpcds & test_execute_tpcds: q_idx -> opt_sql_hash
    tpcds_opt_hash_cache: dict[int, str] = {}
    tpcds_all_passed_flag = False

    def get_preloaded_tables() -> dict:
        nonlocal preloaded_tables_cache
        if preloaded_tables_cache is None:
            if not analysis.touches_executor_or_planner and TABLES_PKL.is_file() and not any(
                "csv.gz" in r for r in analysis.patched_files
            ):
                preloaded_tables_cache = pickle.loads(TABLES_PKL.read_bytes())
            else:
                import csv, gzip
                from sqlglot.executor.table import Table
                from tests.helpers import FIXTURES_DIR, TPCH_SCHEMA, TPCDS_SCHEMA

                def _load(d, schema):
                    out = {}
                    for table, columns in schema.items():
                        fn = f"{d}{table}.csv.gz"
                        with gzip.open(fn, "rt", newline="") as f:
                            reader = csv.reader(f, delimiter="|")
                            next(reader)
                            rows, ctypes = [], []
                            for row in reader:
                                if not ctypes:
                                    for v in row:
                                        try:
                                            ctypes.append(type(ast.literal_eval(v)))
                                        except (ValueError, SyntaxError):
                                            ctypes.append(str)
                                rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
                        out[table] = Table(columns=columns, rows=rows)
                    return out

                preloaded_tables_cache = {
                    "tpch": _load(FIXTURES_DIR + "/optimizer/tpc-h/", TPCH_SCHEMA),
                    "tpcds": _load(FIXTURES_DIR + "/optimizer/tpc-ds/", TPCDS_SCHEMA),
                }
        return preloaded_tables_cache

    def get_duckdb_conn(which: str):
        if which not in duckdb_conns:
            import duckdb
            from tests.helpers import FIXTURES_DIR, TPCH_SCHEMA, TPCDS_SCHEMA
            conn = duckdb.connect()
            schema = TPCH_SCHEMA if which == "tpch" else TPCDS_SCHEMA
            d = FIXTURES_DIR + ("/optimizer/tpc-h/" if which == "tpch" else "/optimizer/tpc-ds/")
            for table, columns in schema.items():
                fn = f"{d}{table}.csv.gz"
                conn.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM READ_CSV('{fn}', delim='|', header=True, columns={columns})"
                )
            duckdb_conns[which] = conn
        return duckdb_conns[which]

    def verify_test_execute_tpch() -> bool:
        from sqlglot import parse_one, find_tables
        from sqlglot.executor import execute
        from tests.helpers import TPCH_SCHEMA, load_sql_fixture_pairs
        sqls = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        for args, sql, _ in sqls:
            expr = parse_one(sql)
            expr = pickle.loads(pickle.dumps(expr))
            if args.get("execute"):
                tbls_all = get_preloaded_tables()["tpch"]
                tbls = {t.name: tbls_all[t.name] for t in find_tables(expr)}
                res_tbl = execute(expr, schema=TPCH_SCHEMA, tables=tbls)
                res_tbl = pickle.loads(pickle.dumps(res_tbl))
                import numpy as np, pandas as pd
                from pandas.testing import assert_frame_equal
                from sqlglot import transpile
                conn = get_duckdb_conn("tpch")
                a = conn.execute(transpile(sql, write="duckdb")[0]).fetchdf()
                b = pd.DataFrame(
                    ((np.nan if c is None else c for c in r) for r in res_tbl.rows),
                    columns=res_tbl.columns,
                )
                assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
        return False

    def verify_test_optimized_tpch() -> bool:
        from sqlglot import transpile
        from tests.helpers import load_sql_fixture_pairs
        sqls = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        csv_touched = any("tpc-h" in r and r.endswith(".csv.gz") for r in analysis.patched_files)
        fix_changed = set(analysis.fixture_changed_indices.get("optimizer/tpc-h/tpc-h.sql", ()))
        for idx, (_, sql, optimized) in enumerate(sqls):
            s_duck = transpile(sql, write="duckdb")[0]
            o_duck = transpile(optimized, write="duckdb")[0]
            if (
                not csv_touched
                and idx not in fix_changed
                and idx < len(maps.tpch_opt_hashes)
                and (_hash_str(s_duck), _hash_str(o_duck)) == maps.tpch_opt_hashes[idx]
            ):
                continue
            from pandas.testing import assert_frame_equal
            conn = get_duckdb_conn("tpch")
            a = conn.execute(s_duck).fetchdf()
            b = conn.execute(o_duck).fetchdf()
            for ci, col in enumerate(b.columns):
                if "_col_" in col:
                    b.rename(columns={col: a.columns[ci]}, inplace=True)
            assert_frame_equal(a, b)
        return False

    def verify_test_tpcds() -> bool:
        nonlocal tpcds_all_passed_flag
        from sqlglot import parse_one, optimizer
        from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs
        sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
        flt = _compute_fixture_filter("optimizer/tpc-ds/tpc-ds.sql", analysis, maps)
        for qi, (_, sql, expected) in enumerate(sqls):
            if flt is not None and qi not in flt:
                continue
            expr = parse_one(sql)
            opt_expr = optimizer.optimize(expr, schema=TPCDS_SCHEMA)
            if analysis.touches_serde:
                opt_expr = pickle.loads(pickle.dumps(opt_expr))
            actual = opt_expr.sql(pretty=True)
            tpcds_opt_hash_cache[qi] = _hash_str(actual)
            if actual != expected:
                return True
        if flt is None or len(flt) == len(sqls):
            tpcds_all_passed_flag = True
        return False

    def verify_test_execute_tpcds() -> bool:
        from sqlglot import parse_one, find_tables, transpile
        from sqlglot.optimizer import optimize
        from sqlglot.executor import execute
        from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

        fix_touched = "optimizer/tpc-ds/tpc-ds.sql" in analysis.patched_files
        csv_touched = any("tpc-ds" in r and r.endswith(".csv.gz") for r in analysis.patched_files)

        # If test_tpcds ALREADY verified all 99 optimized ASTs match baseline verbatim,
        # and neither executor/planner nor duckdb nor serde nor CSVs changed -> GUARANTEED PASS!
        if (
            tpcds_all_passed_flag
            and not analysis.touches_executor_or_planner
            and not analysis.touches_duckdb
            and not analysis.touches_serde
            and not fix_touched
            and not csv_touched
        ):
            return False

        sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
        if analysis.touches_serde:
            for _, sql, _ in sqls:
                e = parse_one(sql)
                _ = pickle.loads(pickle.dumps(e))

        exec_items = []
        qidx_to_eidx = {q["q_idx"]: ei for ei, q in enumerate(maps.tpcds_exec_queries)}

        if not fix_touched and not csv_touched and not analysis.py_class_or_top_files:
            if analysis.touches_executor_or_planner:
                emask = _impacted_bits_from_subcov(maps.tpcds_exec_cov, analysis.py_changed_lines)
            else:
                emask = _impacted_bits_from_subcov(maps.tpcds_opt_cov, analysis.py_changed_lines)
            for ei in _bits_to_list(emask):
                qinfo = maps.tpcds_exec_queries[ei]
                qi = qinfo["q_idx"]
                meta, sql, _ = sqls[qi]
                exec_items.append((ei, qi, meta, sql, qinfo["dur_ms"]))
        else:
            changed_q = set(analysis.fixture_changed_indices.get("optimizer/tpc-ds/tpc-ds.sql", ()))
            for qi, (meta, sql, _) in enumerate(sqls):
                if fix_touched and not analysis.touches_any_py and qi not in changed_q:
                    continue
                if meta.get("execute"):
                    ei = qidx_to_eidx.get(qi)
                    dms = maps.tpcds_exec_queries[ei]["dur_ms"] if ei is not None else 200
                    exec_items.append((ei, qi, meta, sql, dms))

        # Sort fastest-first so failures are caught immediately!
        exec_items.sort(key=lambda x: x[4])

        for ei, qi, meta, sql, _ in exec_items:
            # Stage 1 shortcut using shared tpcds_opt_hash_cache!
            if (
                ei is not None
                and not analysis.touches_executor_or_planner
                and not fix_touched
                and not csv_touched
            ):
                qinfo = maps.tpcds_exec_queries[ei]
                cached_h = tpcds_opt_hash_cache.get(qi)
                if cached_h is None:
                    expr0 = parse_one(sql)
                    if analysis.touches_serde:
                        expr0 = pickle.loads(pickle.dumps(expr0))
                    opt_expr = optimize(expr0, schema=TPCDS_SCHEMA)
                    cached_h = _hash_str(opt_expr.sql(pretty=True))
                    tpcds_opt_hash_cache[qi] = cached_h
                opt_same = (cached_h == qinfo["opt_hash"])
                duck_same = True
                if analysis.touches_duckdb:
                    duck_same = (_hash_str(transpile(sql, write="duckdb")[0]) == qinfo["duckdb_hash"])
                if opt_same and duck_same:
                    continue

            expr = parse_one(sql)
            if analysis.touches_serde:
                expr = pickle.loads(pickle.dumps(expr))

            tbls_all = get_preloaded_tables()["tpcds"]
            tbls = {t.name: tbls_all[t.name] for t in find_tables(expr)}
            res_tbl = execute(expr, schema=TPCDS_SCHEMA, tables=tbls)
            if analysis.touches_serde:
                res_tbl = pickle.loads(pickle.dumps(res_tbl))

            if ei is not None and not fix_touched and not csv_touched and not analysis.touches_duckdb:
                qinfo = maps.tpcds_exec_queries[ei]
                if _normalize_table_hash(res_tbl) == qinfo["table_hash"]:
                    continue
                # Fast structural fail without opening DuckDB (saves ~0.8s!)
                if "columns" in qinfo and (
                    tuple(res_tbl.columns) != qinfo["columns"]
                    or len(res_tbl.rows) != qinfo["nrows"]
                ):
                    return True

            import numpy as np, pandas as pd
            from pandas.testing import assert_frame_equal
            conn = get_duckdb_conn("tpcds")
            a = conn.execute(transpile(sql, write="duckdb")[0]).fetchdf()
            b = pd.DataFrame(
                ((np.nan if c is None else c for c in r) for r in res_tbl.rows),
                columns=res_tbl.columns,
            )
            assert_frame_equal(a, b, check_dtype=False, check_index_type=False)

        if fix_touched:
            for qi in analysis.fixture_changed_indices.get("optimizer/tpc-ds/tpc-ds.sql", ()):
                if qi < len(sqls):
                    _ = parse_one(sqls[qi][1])
        return False

    def verify_test_generator_memory() -> bool:
        from sqlglot.dialects import Dialect
        if not analysis.py_class_or_top_files:
            mask = _impacted_bits_from_subcov(maps.gen_mem_cov, analysis.py_changed_lines)
            dialects_to_check = [maps.gen_mem_dialects[i] for i in _bits_to_list(mask)]
        elif analysis.py_dialect_mods and "sqlglot.dialects.dialect" not in analysis.py_dialect_mods:
            stems = {m.split(".")[-1] for m in analysis.py_dialect_mods}
            dialects_to_check = [d for d in maps.gen_mem_dialects if d in stems or d == ""]
            if not dialects_to_check:
                dialects_to_check = list(maps.gen_mem_dialects)
        else:
            dialects_to_check = list(maps.gen_mem_dialects)

        for name in dialects_to_check:
            dialect = Dialect.get_or_raise(name)
            for _ in range(100):
                dialect.generator()
            gc.collect()
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                for _ in range(1000):
                    dialect.generator()
                gc.collect()
                after = tracemalloc.get_traced_memory()[0]
            finally:
                tracemalloc.stop()
            if not (after - before < 1024):
                return True
        return False

    def verify_test_scope() -> bool:
        """Per-section verifier for TestOptimizer.test_scope (4 independent sections!)."""
        from sqlglot import exp, parse_one
        from sqlglot.optimizer.scope import build_scope, traverse_scope

        if not analysis.py_class_or_top_files and maps.scope_sec_cov:
            smask = _impacted_bits_from_subcov(maps.scope_sec_cov, analysis.py_changed_lines)
        else:
            smask = 0b1111

        if smask & 1:
            ast_node = parse_one("SELECT IF(a IN UNNEST(b), 1, 0) AS c FROM t", dialect="bigquery")
            if build_scope(ast_node).columns != [exp.column("a"), exp.column("b")]:
                return True
        if smask & 8:
            root = build_scope(parse_one("SELECT 1 UNION (SELECT 2 INTERSECT SELECT 3)"))
            if [s.expression.sql() for s in root.set_operation_scopes] != [
                "SELECT 1", "SELECT 2 INTERSECT SELECT 3"
            ]:
                return True
            sql = """
            WITH q AS (SELECT x.b FROM x), r AS (SELECT y.b FROM y),
            z as (SELECT cola, colb FROM (VALUES(1, 'test')) AS tab(cola, colb))
            SELECT r.b, s.b FROM r JOIN (SELECT y.c AS b FROM y) s ON s.b = r.b
            WHERE s.b > (SELECT MAX(x.a) FROM x WHERE x.b = s.b)
            """
            expression = parse_one(sql)
            for scopes in traverse_scope(expression), list(build_scope(expression).traverse()):
                if len(scopes) != 7:
                    return True
                expected_sqls = [
                    "SELECT x.b FROM x",
                    "SELECT y.b FROM y",
                    "(VALUES (1, 'test')) AS tab(cola, colb)",
                    "SELECT cola, colb FROM (VALUES (1, 'test')) AS tab(cola, colb)",
                    "SELECT y.c AS b FROM y",
                    "SELECT MAX(x.a) FROM x WHERE x.b = s.b",
                    parse_one(sql).sql(),
                ]
                for sc, ex_s in zip(scopes, expected_sqls):
                    if sc.expression.sql() != ex_s:
                        return True
        if smask & 4:
            chain = parse_one("SELECT 1" + " UNION SELECT 1 INTERSECT SELECT 1" * 1000, read="hive")
            if len(list(build_scope(chain).traverse())) != 4001:
                return True
        if smask & 2:
            many_unions = parse_one(" UNION ALL ".join(["SELECT x FROM t"] * 10000))
            s1 = list(build_scope(many_unions).traverse())
            s2 = traverse_scope(many_unions)
            if len(s1) != len(s2) or not all(x.expression is y.expression for x, y in zip(s1, s2)):
                return True
        return False

    def verify_test_max_nodes_probe() -> bool | None:
        from sqlglot import exp, parse_one
        from sqlglot.errors import ParseError
        try:
            assert isinstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select)
            cols = [f"col{i}" for i in range(500)]
            assert isinstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=1_000_000), exp.Select)
            expr = "x" + "=x" * 100
            sql_small = f"SELECT {','.join(expr for _ in range(40))}"
            raised = False
            try:
                parse_one(sql_small, max_nodes=5000)
            except ParseError as e:
                assert "(5000)" in str(e) and "Maximum number of AST nodes" in str(e)
                raised = True
            if not raised:
                return True
            raised2 = False
            try:
                parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
            except ParseError:
                raised2 = True
            if not raised2:
                return True
            assert isinstance(parse_one("SELECT 1, 2, 3"), exp.Select)
        except Exception:
            return True
        return None

    def run_single_test(num: int, timeout_s: int) -> bool:
        tid = maps.num_to_test[num]["id"]
        tf, cls_name, meth = tid.split("::")

        try:
            signal.alarm(timeout_s)
            if tf == "tests/test_executor.py":
                if meth == "test_execute_tpch":
                    res = verify_test_execute_tpch()
                    signal.alarm(0)
                    return res
                if meth == "test_optimized_tpch":
                    res = verify_test_optimized_tpch()
                    signal.alarm(0)
                    return res
                if meth == "test_execute_tpcds":
                    res = verify_test_execute_tpcds()
                    signal.alarm(0)
                    return res
            elif tf == "tests/test_generator.py" and meth == "test_generator_memory":
                res = verify_test_generator_memory()
                signal.alarm(0)
                return res
            elif tf == "tests/test_optimizer.py" and meth == "test_tpcds" and "tests/test_optimizer.py" not in analysis.patched_files:
                res = verify_test_tpcds()
                signal.alarm(0)
                return res
            elif tf == "tests/test_optimizer.py" and meth == "test_scope" and "tests/test_optimizer.py" not in analysis.patched_files:
                res = verify_test_scope()
                signal.alarm(0)
                return res
        except (_FastFail, _TestTimeout, Exception):
            signal.alarm(0)
            return True

        mod_name = tf[:-3].replace("/", ".")
        if mod_name not in loaded_modules:
            try:
                signal.alarm(timeout_s)
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "ProcessPoolExecutor"):
                    mod.ProcessPoolExecutor = _BoundPool
                if hasattr(mod, "wait"):
                    mod.wait = _fake_wait
                if hasattr(mod, "as_completed"):
                    mod.as_completed = _fake_as_completed
                loaded_modules[mod_name] = mod
                signal.alarm(0)
            except BaseException as e:
                signal.alarm(0)
                loaded_modules[mod_name] = e
                return True

        mod = loaded_modules[mod_name]
        if isinstance(mod, BaseException):
            return True

        test_cls = getattr(mod, cls_name, None)
        if test_cls is None:
            return True

        cls_key = (tf, cls_name)
        if cls_key not in setup_classes_done:
            try:
                signal.alarm(timeout_s)
                if tf == "tests/test_executor.py":
                    test_cls.cache = {}
                    test_cls.tpch_tables = {}
                    test_cls.tpcds_tables = {}
                    setup_classes_done[cls_key] = True
                else:
                    test_cls.setUpClass()
                    setup_classes_done[cls_key] = True
                signal.alarm(0)
            except BaseException as e:
                signal.alarm(0)
                setup_classes_done[cls_key] = e
                return True

        if isinstance(setup_classes_done[cls_key], BaseException):
            return True

        restore_hooks = []
        if tf == "tests/test_optimizer.py" and meth in OPT_TEST_TO_FIXTURES:
            orig_pairs = getattr(mod, "load_sql_fixture_pairs", None)
            orig_single = getattr(mod, "load_sql_fixtures", None)

            def _make_filtered_pairs(orig_fn):
                def _filtered(filename):
                    flt = _compute_fixture_filter(filename, analysis, maps)
                    if flt is None:
                        yield from orig_fn(filename)
                    else:
                        for i, item in enumerate(orig_fn(filename)):
                            if i in flt:
                                yield item
                return _filtered

            def _make_filtered_single(orig_fn):
                def _filtered(filename):
                    if not analysis.touches_any_py and filename in analysis.fixture_changed_indices:
                        flt = set(analysis.fixture_changed_indices[filename])
                        for i, item in enumerate(orig_fn(filename)):
                            if i in flt:
                                yield item
                    else:
                        yield from orig_fn(filename)
                return _filtered

            if orig_pairs is not None:
                mod.load_sql_fixture_pairs = _make_filtered_pairs(orig_pairs)
                restore_hooks.append((mod, "load_sql_fixture_pairs", orig_pairs))
            if orig_single is not None:
                mod.load_sql_fixtures = _make_filtered_single(orig_single)
                restore_hooks.append((mod, "load_sql_fixtures", orig_single))

        saved_dialects = None
        if tf == "tests/test_dialect_entry_points.py":
            import sqlglot
            saved_dialects = dict(sqlglot.Dialect._classes)

        failed = False
        try:
            signal.alarm(timeout_s)
            case = test_cls(meth)
            result = unittest.TestResult()
            case.run(result)
            signal.alarm(0)
            if not result.wasSuccessful():
                failed = True
        except (_FastFail, _TestTimeout):
            signal.alarm(0)
            failed = True
        except BaseException:
            signal.alarm(0)
            failed = True
        finally:
            for obj, attr, orig in restore_hooks:
                setattr(obj, attr, orig)
            if saved_dialects is not None:
                import sqlglot
                sqlglot.Dialect._classes.update(saved_dialects)

        return failed

    # -----------------------------------------------------------------------
    # Main Verification Loop
    # -----------------------------------------------------------------------
    est_cpu = {n: _estimate_verify_cpu_ms(n, dur_ms_by_num[n], analysis, maps) for n in remaining}
    tests_run = 0
    tests_passed = 0
    tests_failed = 0
    tests_slacked = 0
    tests_gated = 0
    tests_equiv_skipped = 0

    # If pure identity-op equivalence is runtime-proven, run all fast non-benchmark tests FIRST;
    # if ZERO fast tests fail, benchmark tests are mathematically guaranteed equivalent!
    deferred_benchmarks: set[int] = set()
    if identity_equiv_proven:
        for n in list(remaining):
            if maps.num_to_test[n]["id"] in BENCHMARK_TESTS_FOR_EQUIV:
                remaining.remove(n)
                deferred_benchmarks.add(n)

    def _run_loop() -> None:
        nonlocal tests_run, tests_passed, tests_failed, tests_slacked, tests_gated
        while remaining:
            if time.monotonic() > commit_deadline:
                confirmed_fail.update(remaining)
                tests_gated += len(remaining)
                remaining.clear()
                break

            cost_F = optimal_cost_ms(confirmed_fail, dur_ms_by_num)
            cost_FU = optimal_cost_ms(confirmed_fail | remaining, dur_ms_by_num)
            if cost_FU == cost_F:
                tests_slacked += len(remaining)
                confirmed_fail.update(remaining)
                remaining.clear()
                break

            if not confirmed_fail:
                nxt = min(remaining, key=lambda n: (est_cpu[n], dur_ms_by_num[n], n))
                if maps.num_to_test[nxt]["id"] == "tests/test_parser.py::TestParser::test_max_nodes":
                    probe = verify_test_max_nodes_probe()
                    if probe is True:
                        confirmed_fail.add(nxt)
                        remaining.remove(nxt)
                        tests_run += 1
                        tests_failed += 1
                        continue
                    confirmed_fail.add(nxt)
                    remaining.remove(nxt)
                    tests_gated += 1
                    continue
            else:
                best_n = None
                best_gain = 0
                for n in sorted(remaining, key=lambda x: (-dur_ms_by_num[x], est_cpu[x], x)):
                    d_sim = dur_ms_by_num[n]
                    e_cpu = est_cpu[n]
                    if e_cpu >= d_sim and d_sim < 1000:
                        continue
                    m_sim = optimal_cost_ms(confirmed_fail | {n}, dur_ms_by_num) - cost_F
                    gain = m_sim - e_cpu
                    thresh = 0.95 if m_sim >= 1500 else 0.85
                    if gain > best_gain and e_cpu < m_sim * thresh:
                        best_gain = gain
                        best_n = n
                        if gain >= 1000:
                            break

                if best_n is None:
                    rem_cpu_sum = sum(est_cpu[n] for n in remaining)
                    if (cost_FU - cost_F) > rem_cpu_sum * 1.15 and rem_cpu_sum <= 2000:
                        best_n = min(remaining, key=lambda n: (est_cpu[n], -dur_ms_by_num[n], n))
                    else:
                        tests_gated += len(remaining)
                        confirmed_fail.update(remaining)
                        remaining.clear()
                        break

                nxt = best_n
                if maps.num_to_test[nxt]["id"] == "tests/test_parser.py::TestParser::test_max_nodes":
                    probe = verify_test_max_nodes_probe()
                    if probe is True:
                        confirmed_fail.add(nxt)
                        remaining.remove(nxt)
                        tests_run += 1
                        tests_failed += 1
                        continue
                    confirmed_fail.add(nxt)
                    remaining.remove(nxt)
                    tests_gated += 1
                    continue

            remaining.remove(nxt)
            t_left = max(2, int(commit_deadline - time.monotonic()))
            t_lim = min(t_left, max(5, min(25, (dur_ms_by_num[nxt] // 1000) * 3 + 5)))
            failed = run_single_test(nxt, t_lim)
            tests_run += 1
            if failed:
                confirmed_fail.add(nxt)
                tests_failed += 1
            else:
                tests_passed += 1

    _run_loop()

    if deferred_benchmarks:
        if confirmed_fail:
            # Unexpectedly a fast unit test DID fail -> restore benchmarks and verify normally!
            remaining.update(deferred_benchmarks)
            _run_loop()
        else:
            tests_equiv_skipped = len(deferred_benchmarks)

    for conn in duckdb_conns.values():
        try:
            conn.close()
        except Exception:
            pass

    stats = {
        "run": tests_run,
        "pass": tests_passed,
        "fail": tests_failed,
        "slack": tests_slacked,
        "gate": tests_gated,
    }
    if tests_equiv_skipped:
        stats["equiv"] = tests_equiv_skipped
    return sorted(confirmed_fail), stats


def verify_commit_in_fork(
    analysis: CommitAnalysis,
    maps: ImpactMaps,
    dur_ms_by_num: dict[int, int],
    commit_timeout_s: float = 35.0,
) -> tuple[list[int], dict]:
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.close(devnull)
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass

        try:
            with OverlayWorkspace(SQLGLOT_ROOT, analysis.patched_files) as ov_root:
                deadline = time.monotonic() + max(3.0, commit_timeout_s - 1.0)
                selected, stats = _run_child_verification(
                    analysis, maps, dur_ms_by_num, ov_root, deadline
                )
            payload = pickle.dumps((True, selected, stats), protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as e:
            payload = pickle.dumps((False, str(e), {}), protocol=pickle.HIGHEST_PROTOCOL)

        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                written += os.write(w_fd, view[written:])
        except Exception:
            pass
        finally:
            try:
                os.close(w_fd)
            except Exception:
                pass
            os._exit(0)

    os.close(w_fd)
    chunks: list[bytes] = []
    t_end = time.monotonic() + commit_timeout_s + 2.0
    try:
        while True:
            rem = t_end - time.monotonic()
            if rem <= 0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                break
            rlist, _, _ = select.select([r_fd], [], [], min(0.5, rem))
            if rlist:
                chunk = os.read(r_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    raw = b"".join(chunks)
    if raw:
        try:
            ok, selected, stats = pickle.loads(raw)
            if ok:
                return selected, stats
            return list(analysis.candidate_nums), {"err": selected}
        except Exception as e:
            return list(analysis.candidate_nums), {"err": str(e)}

    return list(analysis.candidate_nums), {"err": "child_timeout_or_crash"}


# ---------------------------------------------------------------------------
# Zero-Fork In-Process Verifier for Pure Fixture-Only Optimizer Commits
# ---------------------------------------------------------------------------
class PureFixtureRunner:
    """Loads tests.test_optimizer ONCE in parent and verifies pure .sql optimizer
    fixture commits in ~2-5ms each with zero fork/import overhead."""

    def __init__(self) -> None:
        self._ready = False
        self._mod = None
        self._cls = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        # Temporarily suppress stdout during import/setup
        orig_out = sys.stdout
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        try:
            class _NoPicklePool(FastSyncPool):
                def __init__(self, *a, **kw):
                    super().__init__(*a, do_pickle=False, **kw)

            def _fake_wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
                return (set(fs), set())

            def _fake_as_completed(fs, timeout=None):
                return iter(list(fs))

            cf.ProcessPoolExecutor = _NoPicklePool
            cf.wait = _fake_wait
            cf.as_completed = _fake_as_completed

            import tests.test_optimizer as topt
            topt.ProcessPoolExecutor = _NoPicklePool
            topt.wait = _fake_wait
            topt.as_completed = _fake_as_completed
            topt.TestOptimizer.setUpClass()
            self._mod = topt
            self._cls = topt.TestOptimizer
            self._ready = True
        finally:
            sys.stdout = orig_out
            devnull.close()

    def can_handle(self, analysis: CommitAnalysis, maps: ImpactMaps) -> bool:
        if analysis.touches_any_py or not analysis.candidate_nums:
            return False
        # Only handle pure optimizer .sql fixture commits (not CSVs, not tpch/tpcds)
        for rel, content in analysis.patched_files.items():
            if content is None:
                return False
            if not (rel.startswith("tests/fixtures/optimizer/") and rel.endswith(".sql")):
                return False
            if "tpc-ds" in rel or "tpc-h" in rel:
                return False
        for n in analysis.candidate_nums:
            if maps.num_to_test[n]["file"] != "tests/test_optimizer.py":
                return False
        return True

    def verify(self, analysis: CommitAnalysis, maps: ImpactMaps) -> tuple[list[int], dict]:
        self._ensure_ready()
        mod = self._mod
        cls = self._cls
        orig_pairs = mod.load_sql_fixture_pairs
        orig_single = mod.load_sql_fixtures
        orig_subtest = unittest.TestCase.subTest
        orig_stdout = sys.stdout

        @contextlib.contextmanager
        def _fast_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
            try:
                yield
            except unittest.SkipTest:
                pass
            except Exception as e:
                raise _FastFail(e) from e

        def _mem_pairs(filename):
            rel = f"tests/fixtures/{filename}"
            if rel in analysis.patched_files:
                pairs = parse_sql_fixture_pairs_text(analysis.patched_files[rel])
                flt = set(analysis.fixture_changed_indices.get(filename, ()))
                for i, item in enumerate(pairs):
                    if not flt or i in flt:
                        yield item
            else:
                return

        def _mem_single(filename):
            rel = f"tests/fixtures/{filename}"
            if rel in analysis.patched_files:
                items = parse_sql_fixtures_text(analysis.patched_files[rel])
                flt = set(analysis.fixture_changed_indices.get(filename, ()))
                for i, item in enumerate(items):
                    if not flt or i in flt:
                        yield item
            else:
                return

        confirmed: list[int] = []
        run_cnt = 0
        devnull = open(os.devnull, "w")
        try:
            sys.stdout = devnull
            mod.load_sql_fixture_pairs = _mem_pairs
            mod.load_sql_fixtures = _mem_single
            unittest.TestCase.subTest = _fast_subtest

            for n in analysis.candidate_nums:
                meth = maps.num_to_test[n]["id"].split("::")[2]
                run_cnt += 1
                failed = False
                try:
                    case = cls(meth)
                    result = unittest.TestResult()
                    case.run(result)
                    if not result.wasSuccessful():
                        failed = True
                except (_FastFail, BaseException):
                    failed = True
                if failed:
                    confirmed.append(n)
        finally:
            sys.stdout = orig_stdout
            devnull.close()
            mod.load_sql_fixture_pairs = orig_pairs
            mod.load_sql_fixtures = orig_single
            unittest.TestCase.subTest = orig_subtest

        return confirmed, {"run": run_cnt, "fail": len(confirmed), "fast_fix": True}
