#!/usr/bin/env python3
"""Offline build-time map generator for Blast Radius (V3).

Produces /out/maps.zlib containing:
  1. Per-test dynamic line coverage (Python 3.13 sys.monitoring + DISABLE)
  2. Per-test dialect MRO usage & Expression subclass usage
  3. Per-test TrackedDict key-level usage (FUNCTIONS, SHOW_PARSERS, NUMERIC_ESCAPES,
     STATEMENT_PARSERS, FUNCTION_PARSERS, PROPERTY_PARSERS, TYPE_MAPPING,
     PROPERTIES_LOCATION, EXPRESSION_METADATA)
  4. Per-test file I/O reads + eager/lazy import graph
  5. Per-subtest coverage & baseline snapshots for top slow tests:
     - test_execute_tpcds (57 queries: opt/exec cov + baseline AST, cols, nrows, Table hash)
     - test_optimized_tpch (22 queries: baseline DuckDB SQL hashes)
     - test_generator_memory (per dialect)
     - test_scope (per section: 4 sections!)
     - 14 slow test_optimizer.py fixture tests (0 errors with full TestOptimizer.setUp schema!)
"""

from __future__ import annotations

import ast
import builtins
import collections
import concurrent.futures as cf
import csv
import gzip
import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
import time
import types
import unittest
import zlib
from functools import partial
from pathlib import Path

os.environ["SKIP_INTEGRATION"] = "0"

SQLGLOT_ROOT = Path("/app/sqlglot").resolve()
if str(SQLGLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SQLGLOT_ROOT))

TESTS_JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/tests.json")
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/maps.zlib")

TRACKED_DICT_ATTRS = (
    "FUNCTIONS",
    "SHOW_PARSERS",
    "STATEMENT_PARSERS",
    "FUNCTION_PARSERS",
    "PROPERTY_PARSERS",
    "NUMERIC_ESCAPES",
    "TYPE_MAPPING",
    "PROPERTIES_LOCATION",
    "EXPRESSION_METADATA",
)


def rel_repo_path(filepath: str | os.PathLike | None) -> str | None:
    if not filepath:
        return None
    try:
        p = Path(filepath).resolve()
        return p.relative_to(SQLGLOT_ROOT).as_posix()
    except Exception:
        return None


class _SyncFuture:
    def __init__(self, value=None, exc=None):
        self._value = value
        self._exc = exc

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._value

    def exception(self, timeout=None):
        return self._exc

    def done(self):
        return True

    def add_done_callback(self, fn):
        fn(self)


class SyncPoolExecutor:
    def __init__(self, *args, initializer=None, initargs=(), **kwargs):
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
            p_args = pickle.loads(pickle.dumps(args))
            p_kwargs = pickle.loads(pickle.dumps(kwargs)) if kwargs else {}
            res = fn(*p_args, **p_kwargs)
            res = pickle.loads(pickle.dumps(res))
            return _SyncFuture(value=res)
        except Exception as e:
            return _SyncFuture(exc=e)


def build_static_ast_info() -> dict:
    py_files = sorted(
        p for p in SQLGLOT_ROOT.rglob("*.py")
        if p.relative_to(SQLGLOT_ROOT).parts[0] in ("sqlglot", "tests", "sqlglotc")
    )
    mod_to_rel: dict[str, str] = {}
    for p in py_files:
        rel = p.relative_to(SQLGLOT_ROOT).as_posix()
        if rel.endswith("/__init__.py"):
            mod = rel[:-len("/__init__.py")].replace("/", ".")
        else:
            mod = rel[:-3].replace("/", ".")
        mod_to_rel[mod] = rel

    sym_ref_lines: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    py_file_list: list[str] = []

    for p in py_files:
        rel = p.relative_to(SQLGLOT_ROOT).as_posix()
        py_file_list.append(rel)
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        seen_in_file: set[tuple[str, int]] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                key = (node.id, node.lineno)
                if key not in seen_in_file:
                    seen_in_file.add(key)
                    sym_ref_lines[node.id].append((rel, node.lineno))
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                key = (node.attr, node.lineno)
                if key not in seen_in_file:
                    seen_in_file.add(key)
                    sym_ref_lines[node.attr].append((rel, node.lineno))

    return {
        "mod_to_rel": mod_to_rel,
        "py_files": py_file_list,
        "sym_ref_lines": dict(sym_ref_lines),
    }


def collect_eager_imports(test_files: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    code = (
        "import json, sys, pathlib\n"
        "root = pathlib.Path('/app/sqlglot').resolve()\n"
        "import sqlglot\n"
        "mods = []\n"
        "for m in list(sys.modules.values()):\n"
        "    f = getattr(m, '__file__', None)\n"
        "    if f:\n"
        "        try: mods.append(pathlib.Path(f).resolve().relative_to(root).as_posix())\n"
        "        except Exception: pass\n"
        "print(json.dumps(sorted(set(mods))))\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    core_sqlglot_mods = json.loads(out)

    eager_by_test_file: dict[str, list[str]] = {}
    for tf in test_files:
        mod_name = tf[:-3].replace("/", ".")
        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r_fd)
            try:
                importlib.import_module(mod_name)
                mods = set()
                for m in list(sys.modules.values()):
                    rel = rel_repo_path(getattr(m, "__file__", None))
                    if rel:
                        mods.add(rel)
                data = json.dumps(sorted(mods)).encode()
            except Exception as e:
                data = json.dumps({"error": str(e)}).encode()
            os.write(w_fd, data)
            os.close(w_fd)
            os._exit(0)
        else:
            os.close(w_fd)
            chunks = []
            while True:
                chunk = os.read(r_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            os.close(r_fd)
            os.waitpid(pid, 0)
            res = json.loads(b"".join(chunks).decode())
            eager_by_test_file[tf] = res if isinstance(res, list) else []
    return core_sqlglot_mods, eager_by_test_file


def _dict_key_repr(k) -> str:
    if isinstance(k, str):
        return f"S:{k}"
    if isinstance(k, type):
        return f"T:{k.__name__}"
    # TokenType enum etc.
    cls_name = type(k).__name__
    if hasattr(k, "name"):
        return f"E:{cls_name}.{k.name}"
    return f"R:{k!r}"


def run_dynamic_profiling(tests: list[dict], core_sqlglot_mods: list[str]) -> dict:
    t0 = time.perf_counter()
    test_ids = [t["id"] for t in tests]
    id_to_idx = {tid: i for i, tid in enumerate(test_ids)}
    n_tests = len(tests)

    orig_ppe = cf.ProcessPoolExecutor
    orig_wait = cf.wait
    orig_as_completed = cf.as_completed

    def fake_wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
        return (set(fs), set())

    def fake_as_completed(fs, timeout=None):
        return iter(list(fs))

    cf.ProcessPoolExecutor = SyncPoolExecutor
    cf.wait = fake_wait
    cf.as_completed = fake_as_completed

    import sqlglot
    from sqlglot.dialects.dialect import Dialect, Dialects
    from sqlglot.expressions.core import Expr
    from sqlglot.parser import Parser
    from sqlglot.generator import Generator
    from sqlglot.tokens import Tokenizer

    # Current test tracking state
    cur_lines: dict[str, set[int]] = collections.defaultdict(set)
    cur_dialects: set[str] = set()
    cur_exprs: set[str] = set()
    cur_files: set[str] = set()
    cur_dict_keys: set[tuple[str, str, str]] = set()
    track_exprs_enabled = True
    recording_active = False

    # Accumulated maps
    line_cov: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    dialect_usage: dict[str, int] = collections.defaultdict(int)
    expr_usage: dict[str, int] = collections.defaultdict(int)
    file_reads: dict[str, int] = collections.defaultdict(int)
    lazy_imports: dict[str, int] = collections.defaultdict(int)
    # (owner_cls_key, attr_name) -> {key_repr: bitmask}
    dict_key_cov: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    wrapped_dict_owners: set[tuple[str, str]] = set()
    wrapped_classes: set[type] = set()

    class TrackedDict(dict):
        __slots__ = ("_owner", "_attr")

        def __init__(self, orig: dict, owner: str, attr: str):
            super().__init__(orig)
            self._owner = owner
            self._attr = attr

        def __getitem__(self, key):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, _dict_key_repr(key)))
            return super().__getitem__(key)

        def get(self, key, default=None):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, _dict_key_repr(key)))
            return super().get(key, default)

        def __contains__(self, key):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, _dict_key_repr(key)))
            return super().__contains__(key)

        def __iter__(self):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, "*"))
            return super().__iter__()

        def keys(self):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, "*"))
            return super().keys()

        def values(self):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, "*"))
            return super().values()

        def items(self):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, "*"))
            return super().items()

        def copy(self):
            if recording_active:
                cur_dict_keys.add((self._owner, self._attr, "*"))
            return super().copy()

    def wrap_class_dicts(cls: type) -> None:
        if cls in wrapped_classes:
            return
        wrapped_classes.add(cls)
        mod = getattr(cls, "__module__", "")
        if not mod.startswith("sqlglot."):
            return
        owner = f"{mod}:{cls.__qualname__}"
        for attr in TRACKED_DICT_ATTRS:
            val = cls.__dict__.get(attr)
            if type(val) is dict:
                td = TrackedDict(val, owner, attr)
                setattr(cls, attr, td)
                wrapped_dict_owners.add((owner, attr))

    def wrap_dialect_hierarchy(dcls: type) -> None:
        for c in dcls.__mro__:
            wrap_class_dicts(c)
        for sub_attr in ("parser_class", "generator_class", "tokenizer_class"):
            sub = getattr(dcls, sub_attr, None)
            if isinstance(sub, type):
                for c in sub.__mro__:
                    wrap_class_dicts(c)

    # Pre-wrap base classes
    for base in (Dialect, Parser, Generator, Tokenizer):
        wrap_class_dicts(base)

    # Hook Dialect.get_or_raise and Dialect.__init__
    orig_get_or_raise = Dialect.get_or_raise.__func__
    dialect_mro_cache: dict[type, tuple[str, ...]] = {}

    def _dialect_keys(cls: type) -> tuple[str, ...]:
        keys = dialect_mro_cache.get(cls)
        if keys is not None:
            return keys
        wrap_dialect_hierarchy(cls)
        out = set()
        for c in cls.__mro__:
            mod = getattr(c, "__module__", "")
            if mod.startswith("sqlglot."):
                out.add(f"{mod}:{c.__qualname__}")
                out.add(mod)
        for attr in ("parser_class", "generator_class", "tokenizer_class"):
            sub = getattr(cls, attr, None)
            if isinstance(sub, type):
                for c in sub.__mro__:
                    mod = getattr(c, "__module__", "")
                    if mod.startswith("sqlglot."):
                        out.add(f"{mod}:{c.__qualname__}")
                        out.add(mod)
        keys = tuple(out)
        dialect_mro_cache[cls] = keys
        return keys

    @classmethod
    def tracked_get_or_raise(cls, dialect=None):
        res = orig_get_or_raise(cls, dialect)
        if res is not None:
            dk = _dialect_keys(type(res))
            if recording_active:
                cur_dialects.update(dk)
        return res

    Dialect.get_or_raise = tracked_get_or_raise

    orig_dialect_init = Dialect.__init__

    def tracked_dialect_init(self, *args, **kwargs):
        dk = _dialect_keys(type(self))
        if recording_active:
            cur_dialects.update(dk)
        return orig_dialect_init(self, *args, **kwargs)

    Dialect.__init__ = tracked_dialect_init

    # Hook Expr.__init__
    orig_expr_init = Expr.__init__

    def tracked_expr_init(self, **args):
        if recording_active and track_exprs_enabled:
            cname = type(self).__name__
            if cname not in cur_exprs:
                cur_exprs.add(cname)
        return orig_expr_init(self, **args)

    Expr.__init__ = tracked_expr_init

    # Hook open / gzip.open
    orig_open = builtins.open
    orig_gzip_open = gzip.open

    def tracked_open(file, mode="r", *args, **kwargs):
        if recording_active and isinstance(file, (str, os.PathLike)):
            rel = rel_repo_path(file)
            if rel:
                cur_files.add(rel)
        return orig_open(file, mode, *args, **kwargs)

    def tracked_gzip_open(filename, mode="rb", *args, **kwargs):
        if recording_active and isinstance(filename, (str, os.PathLike)):
            rel = rel_repo_path(filename)
            if rel:
                cur_files.add(rel)
        return orig_gzip_open(filename, mode, *args, **kwargs)

    builtins.open = tracked_open
    gzip.open = tracked_gzip_open

    # Setup Python 3.13 sys.monitoring
    mon = sys.monitoring
    TOOL_ID = 2
    mon.use_tool_id(TOOL_ID, "blast_cov")
    code_rel_cache: dict[types.CodeType, str | None] = {}
    root_prefix = str(SQLGLOT_ROOT) + os.sep

    def on_line(code: types.CodeType, line_number: int):
        rel = code_rel_cache.get(code)
        if rel is None and code not in code_rel_cache:
            fn = code.co_filename
            if fn.startswith(root_prefix):
                rel = fn[len(root_prefix):]
            else:
                rel = rel_repo_path(fn)
            code_rel_cache[code] = rel
        if rel:
            cur_lines[rel].add(line_number)
        return mon.DISABLE

    mon.register_callback(TOOL_ID, mon.events.LINE, on_line)

    # Group tests by (file, class_name)
    by_file_class: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for t in tests:
        f, cls_name, meth = t["id"].split("::")
        by_file_class[(f, cls_name)].append(t)

    failures: list[str] = []
    for (tf, cls_name), class_tests in sorted(by_file_class.items()):
        mod_name = tf[:-3].replace("/", ".")
        mod = importlib.import_module(mod_name)
        test_cls = getattr(mod, cls_name)

        if hasattr(mod, "ProcessPoolExecutor"):
            mod.ProcessPoolExecutor = SyncPoolExecutor
        if hasattr(mod, "wait"):
            mod.wait = fake_wait
        if hasattr(mod, "as_completed"):
            mod.as_completed = fake_as_completed

        # Profile setUpClass
        cur_lines = collections.defaultdict(set)
        cur_dialects = set()
        cur_exprs = set()
        cur_files = set()
        cur_dict_keys = set()
        recording_active = True
        track_exprs_enabled = True
        mon.restart_events()
        mon.set_events(TOOL_ID, mon.events.LINE)
        try:
            test_cls.setUpClass()
        finally:
            mon.set_events(TOOL_ID, 0)
            recording_active = False

        setup_lines = {k: set(v) for k, v in cur_lines.items()}
        setup_dialects = set(cur_dialects)
        setup_exprs = set(cur_exprs)
        setup_files = set(cur_files)
        setup_dkeys = set(cur_dict_keys)

        class_bitmask = 0
        for t in class_tests:
            idx = id_to_idx[t["id"]]
            bit = 1 << idx
            class_bitmask |= bit
            meth = t["id"].split("::")[2]

            mods_before = set(sys.modules.keys())
            cur_lines = collections.defaultdict(set)
            cur_dialects = set()
            cur_exprs = set()
            cur_files = set()
            cur_dict_keys = set()

            is_max_nodes = (tf == "tests/test_parser.py" and meth == "test_max_nodes")
            track_exprs_enabled = not is_max_nodes
            recording_active = True
            mon.restart_events()
            mon.set_events(TOOL_ID, mon.events.LINE)

            case = test_cls(meth)
            result = unittest.TestResult()
            saved_dialect_classes = dict(Dialect._classes) if tf == "tests/test_dialect_entry_points.py" else None
            case.run(result)
            if saved_dialect_classes is not None:
                Dialect._classes.update(saved_dialect_classes)

            mon.set_events(TOOL_ID, 0)
            recording_active = False
            track_exprs_enabled = True

            if not result.wasSuccessful():
                failures.append(f"{t['id']}: errs={len(result.errors)} fails={len(result.failures)}")

            if is_max_nodes:
                cur_exprs.update(["Select", "EQ", "Column", "Identifier", "Table", "Literal", "Star", "From"])

            if meth == "test_lazy_load":
                for rel in core_sqlglot_mods:
                    lazy_imports[rel] |= bit

            for mname in set(sys.modules.keys()) - mods_before:
                mobj = sys.modules.get(mname)
                rel = rel_repo_path(getattr(mobj, "__file__", None))
                if rel:
                    lazy_imports[rel] |= bit

            for rel, lns in cur_lines.items():
                d = line_cov[rel]
                for ln in lns:
                    d[ln] |= bit
            for dk in cur_dialects:
                dialect_usage[dk] |= bit
            for ek in cur_exprs:
                expr_usage[ek] |= bit
            for fk in cur_files:
                file_reads[fk] |= bit
            for owner, attr, krep in cur_dict_keys:
                dict_key_cov[(owner, attr)][krep] |= bit

        for rel, lns in setup_lines.items():
            d = line_cov[rel]
            for ln in lns:
                d[ln] |= class_bitmask
        for dk in setup_dialects:
            dialect_usage[dk] |= class_bitmask
        for ek in setup_exprs:
            expr_usage[ek] |= class_bitmask
        for fk in setup_files:
            file_reads[fk] |= class_bitmask
        for owner, attr, krep in setup_dkeys:
            dict_key_cov[(owner, attr)][krep] |= class_bitmask

        try:
            test_cls.tearDownClass()
        except Exception:
            pass

    print(
        f"[phase B] Profiled {n_tests} tests in {time.perf_counter() - t0:.2f}s, "
        f"failures={len(failures)}, wrapped_dicts={len(wrapped_dict_owners)}",
        file=sys.stderr,
    )
    if failures:
        print("  FAILURES:", failures[:10], file=sys.stderr)

    # Restore TrackedDicts to plain dicts before Phase C & pickle
    for cls in wrapped_classes:
        for attr in TRACKED_DICT_ATTRS:
            val = cls.__dict__.get(attr)
            if isinstance(val, TrackedDict):
                setattr(cls, attr, dict(val))

    # Phase C: Subtest-Level Coverage & Baseline Snapshots
    t1 = time.perf_counter()
    subtest_data = collect_subtest_maps(mon, TOOL_ID, code_rel_cache, root_prefix)
    mon.set_events(TOOL_ID, 0)
    mon.free_tool_id(TOOL_ID)

    # Restore hooks
    Dialect.get_or_raise = classmethod(orig_get_or_raise)
    Dialect.__init__ = orig_dialect_init
    Expr.__init__ = orig_expr_init
    builtins.open = orig_open
    gzip.open = orig_gzip_open
    cf.ProcessPoolExecutor = orig_ppe
    cf.wait = orig_wait
    cf.as_completed = orig_as_completed

    # Complete MRO & subclass maps
    for d in Dialects:
        try:
            Dialect.get_or_raise(d)
        except Exception:
            pass

    def build_subclass_map(base_cls: type) -> dict[str, list[str]]:
        all_subs = set()
        stack = [base_cls]
        while stack:
            c = stack.pop()
            if c in all_subs:
                continue
            all_subs.add(c)
            stack.extend(c.__subclasses__())
        res: dict[str, list[str]] = {}
        for c in all_subs:
            mod = getattr(c, "__module__", "")
            if not mod.startswith("sqlglot."):
                continue
            ck = f"{mod}:{c.__qualname__}"
            desc = []
            for s in all_subs:
                if issubclass(s, c):
                    sm = getattr(s, "__module__", "")
                    if sm.startswith("sqlglot."):
                        desc.append(f"{sm}:{s.__qualname__}")
            res[ck] = sorted(set(desc))
        return res

    subclass_maps = {}
    for base in (Dialect, Parser, Generator, Tokenizer):
        subclass_maps.update(build_subclass_map(base))

    expr_subclasses: dict[str, list[str]] = {}
    all_exprs = set()
    stack = [Expr]
    while stack:
        c = stack.pop()
        if c in all_exprs:
            continue
        all_exprs.add(c)
        stack.extend(c.__subclasses__())
    for c in all_exprs:
        expr_subclasses[c.__name__] = sorted(s.__name__ for s in all_exprs if issubclass(s, c))

    # Explicit reads: sqlglot/typing/*.py -> test_typing.py; sqlglotc/* -> test_sqlglotc.py
    typing_tid = "tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata"
    if typing_tid in id_to_idx:
        tbit = 1 << id_to_idx[typing_tid]
        for p in (SQLGLOT_ROOT / "sqlglot/typing").glob("*.py"):
            file_reads[p.relative_to(SQLGLOT_ROOT).as_posix()] |= tbit

    sqlglotc_mask = 0
    for tid, idx in id_to_idx.items():
        if tid.startswith("tests/test_sqlglotc.py::"):
            sqlglotc_mask |= (1 << idx)
    if sqlglotc_mask:
        for p in (SQLGLOT_ROOT / "sqlglotc").rglob("*"):
            if p.is_file():
                file_reads[p.relative_to(SQLGLOT_ROOT).as_posix()] |= sqlglotc_mask

    print(f"[phase C] Subtests & MRO finished in {time.perf_counter() - t1:.2f}s", file=sys.stderr)

    return {
        "test_ids": test_ids,
        "line_cov": {k: dict(v) for k, v in line_cov.items()},
        "dialect_usage": dict(dialect_usage),
        "expr_usage": dict(expr_usage),
        "dict_key_cov": {k: dict(v) for k, v in dict_key_cov.items()},
        "wrapped_dict_owners": sorted(wrapped_dict_owners),
        "file_reads": dict(file_reads),
        "lazy_imports": dict(lazy_imports),
        "subclass_maps": subclass_maps,
        "expr_subclasses": expr_subclasses,
        "subtests": subtest_data,
    }


def collect_subtest_maps(mon, tool_id, code_rel_cache, root_prefix) -> dict:
    from sqlglot import exp, find_tables, parse_one, transpile, optimizer
    from sqlglot.dialects.dialect import Dialect, Dialects
    from sqlglot.executor import execute
    from sqlglot.executor.table import Table
    from sqlglot.optimizer.annotate_types import annotate_types
    from sqlglot.optimizer.scope import build_scope, traverse_scope
    from tests.helpers import FIXTURES_DIR, TPCH_SCHEMA, TPCDS_SCHEMA, load_sql_fixture_pairs, string_to_bool
    import tests.test_optimizer as topt

    sub_lines: dict[str, set[int]] = collections.defaultdict(set)

    def sub_on_line(code: types.CodeType, line_number: int):
        rel = code_rel_cache.get(code)
        if rel is None and code not in code_rel_cache:
            fn = code.co_filename
            if fn.startswith(root_prefix):
                rel = fn[len(root_prefix):]
            else:
                rel = rel_repo_path(fn)
            code_rel_cache[code] = rel
        if rel:
            sub_lines[rel].add(line_number)
        return mon.DISABLE

    mon.register_callback(tool_id, mon.events.LINE, sub_on_line)

    def hash_str(s: str) -> str:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()

    def normalize_table_for_hash(table: Table) -> bytes:
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

    # 1. Load TPC-DS tables
    tpcds_tables: dict[str, Table] = {}
    for table, columns in TPCDS_SCHEMA.items():
        fn = f"{FIXTURES_DIR}/optimizer/tpc-ds/{table}.csv.gz"
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
        tpcds_tables[table] = Table(columns=columns, rows=rows)

    tpcds_sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    tpcds_exec_queries = []
    tpcds_opt_cov: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    tpcds_exec_cov: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    tpcds_all_opt_cov: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    for q_idx, (meta, sql, expected) in enumerate(tpcds_sqls):
        qbit = 1 << q_idx
        sub_lines = collections.defaultdict(set)
        mon.restart_events()
        mon.set_events(tool_id, mon.events.LINE)
        expr = parse_one(sql)
        opt_expr = optimizer.optimize(expr, schema=TPCDS_SCHEMA)
        mon.set_events(tool_id, 0)

        for rel, lns in sub_lines.items():
            d_all = tpcds_all_opt_cov[rel]
            for ln in lns:
                d_all[ln] |= qbit

        if meta.get("execute"):
            e_idx = len(tpcds_exec_queries)
            ebit = 1 << e_idx
            for rel, lns in sub_lines.items():
                d_opt = tpcds_opt_cov[rel]
                for ln in lns:
                    d_opt[ln] |= ebit

            sub_lines = collections.defaultdict(set)
            t_s = time.perf_counter()
            mon.restart_events()
            mon.set_events(tool_id, mon.events.LINE)
            expr2 = parse_one(sql)
            tbls = {t.name: tpcds_tables[t.name] for t in find_tables(expr2)}
            res_table = execute(expr2, schema=TPCDS_SCHEMA, tables=tbls)
            mon.set_events(tool_id, 0)
            dur_ms = int((time.perf_counter() - t_s) * 1000)

            for rel, lns in sub_lines.items():
                d_ex = tpcds_exec_cov[rel]
                for ln in lns:
                    d_ex[ln] |= ebit

            tpcds_exec_queries.append({
                "q_idx": q_idx,
                "dur_ms": dur_ms,
                "opt_hash": hash_str(opt_expr.sql(pretty=True)),
                "duckdb_hash": hash_str(transpile(sql, write="duckdb")[0]),
                "table_hash": normalize_table_for_hash(res_table),
                "columns": tuple(res_table.columns),
                "nrows": len(res_table.rows),
            })

    # 2. test_optimized_tpch hashes
    tpch_sqls = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
    tpch_opt_hashes = [
        (hash_str(transpile(sql, write="duckdb")[0]), hash_str(transpile(opt, write="duckdb")[0]))
        for _, sql, opt in tpch_sqls
    ]

    # 3. test_generator_memory per-dialect coverage
    gen_mem_dialects = [d.value for d in Dialects]
    gen_mem_cov: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for d_idx, d_name in enumerate(gen_mem_dialects):
        dbit = 1 << d_idx
        sub_lines = collections.defaultdict(set)
        mon.restart_events()
        mon.set_events(tool_id, mon.events.LINE)
        dialect = Dialect.get_or_raise(d_name)
        dialect.generator()
        mon.set_events(tool_id, 0)
        for rel, lns in sub_lines.items():
            d = gen_mem_cov[rel]
            for ln in lns:
                d[ln] |= dbit

    # 4. test_scope per-section coverage (4 independent sections!)
    scope_sec_cov: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    def _rec_scope_sec(sec_idx: int, fn):
        nonlocal sub_lines
        sbit = 1 << sec_idx
        sub_lines = collections.defaultdict(set)
        mon.restart_events()
        mon.set_events(tool_id, mon.events.LINE)
        fn()
        mon.set_events(tool_id, 0)
        for rel, lns in sub_lines.items():
            dd = scope_sec_cov[rel]
            for ln in lns:
                dd[ln] |= sbit

    def _scope_s0():
        a = parse_one("SELECT IF(a IN UNNEST(b), 1, 0) AS c FROM t", dialect="bigquery")
        _ = build_scope(a).columns

    def _scope_s1():
        many_unions = parse_one(" UNION ALL ".join(["SELECT x FROM t"] * 20))
        _ = list(build_scope(many_unions).traverse())
        _ = traverse_scope(many_unions)

    def _scope_s2():
        chain = parse_one("SELECT 1" + " UNION SELECT 1 INTERSECT SELECT 1" * 10, read="hive")
        _ = list(build_scope(chain).traverse())

    def _scope_s3():
        root = build_scope(parse_one("SELECT 1 UNION (SELECT 2 INTERSECT SELECT 3)"))
        _ = [s.expression.sql() for s in root.set_operation_scopes]
        sql = """
        WITH q AS (SELECT x.b FROM x), r AS (SELECT y.b FROM y),
        z as (SELECT cola, colb FROM (VALUES(1, 'test')) AS tab(cola, colb))
        SELECT r.b, s.b FROM r JOIN (SELECT y.c AS b FROM y) s ON s.b = r.b
        WHERE s.b > (SELECT MAX(x.a) FROM x WHERE x.b = s.b)
        """
        e = parse_one(sql)
        for sc in traverse_scope(e), list(build_scope(e).traverse()):
            _ = [x.expression.sql() for x in sc]

    for si, sfn in enumerate((_scope_s0, _scope_s1, _scope_s2, _scope_s3)):
        _rec_scope_sec(si, sfn)

    # 5. Per-fixture-entry coverage for slow test_optimizer.py fixture tests (using exact TestOptimizer.setUp schema!)
    topt_inst = topt.TestOptimizer("test_optimize")
    topt_inst.setUp()
    base_schema = topt_inst.schema

    def _run_opt_fixture_cov(
        fixture_file: str,
        fn,
        base_kwargs: dict,
        is_annotate_funcs: bool = False,
        pretty: bool = False,
        execute_default: bool = False,
    ):
        nonlocal sub_lines
        cov: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        dials: dict[str, int] = collections.defaultdict(int)
        pairs = list(load_sql_fixture_pairs(fixture_file))
        errs = 0
        for idx, (meta, sql, expected) in enumerate(pairs):
            bit = 1 << idx
            sub_lines = collections.defaultdict(set)
            mon.restart_events()
            mon.set_events(tool_id, mon.events.LINE)
            try:
                if is_annotate_funcs:
                    dialect_str = meta.get("dialect") or ""
                    sql_q = f"SELECT {sql} FROM tbl"
                    for d_name in dialect_str.split(", "):
                        res = topt.parse_and_optimize(fn, sql_q, d_name, dialect=d_name, **base_kwargs)
                        _ = res.type.sql(d_name)
                        _ = exp.DataType.build(expected, dialect=d_name).sql(d_name)
                        d_obj = Dialect.get_or_raise(d_name)
                        for c in type(d_obj).__mro__:
                            m = getattr(c, "__module__", "")
                            if m.startswith("sqlglot."):
                                dials[m] |= bit
                                dials[f"{m}:{c.__qualname__}"] |= bit
                else:
                    d_name = meta.get("dialect")
                    fk = dict(base_kwargs)
                    if schema := meta.get("schema"):
                        fk["schema"] = json.loads(schema)
                    if (lti := meta.get("leave_tables_isolated")) is not None:
                        fk["leave_tables_isolated"] = string_to_bool(lti)
                    if (vqc := meta.get("validate_qualify_columns")) is not None:
                        fk["validate_qualify_columns"] = string_to_bool(vqc)
                    if d_name:
                        fk["dialect"] = d_name
                    if (cta := meta.get("canonicalize_table_aliases")) is not None:
                        fk["canonicalize_table_aliases"] = string_to_bool(cta)
                    res = topt.parse_and_optimize(fn, sql, d_name, **fk)
                    _ = res.sql(pretty=pretty, dialect=d_name)
                    if execute_default or string_to_bool(meta.get("execute")):
                        _ = parse_one(sql, read=d_name).sql(dialect="duckdb")
                        _ = res.sql(dialect="duckdb")
                    d_obj = Dialect.get_or_raise(d_name)
                    for c in type(d_obj).__mro__:
                        m = getattr(c, "__module__", "")
                        if m.startswith("sqlglot."):
                            dials[m] |= bit
                            dials[f"{m}:{c.__qualname__}"] |= bit
            except Exception as e:
                errs += 1
                if errs <= 3:
                    print(f"[warn] subcov {fixture_file}[{idx}]: {e}", file=sys.stderr)
            finally:
                mon.set_events(tool_id, 0)
            for rel, lns in sub_lines.items():
                dd = cov[rel]
                for ln in lns:
                    dd[ln] |= bit
        print(f"[subcov] {fixture_file}: {len(pairs)} pairs, {len(cov)} files, errs={errs}", file=sys.stderr)
        return {
            "count": len(pairs),
            "cov": {k: dict(v) for k, v in cov.items()},
            "dialects": dict(dials),
        }

    annotate_funcs_schema = {
        "tbl": {
            "bin_col": "BINARY", "str_col": "STRING", "bignum_col": "BIGNUMERIC",
            "date_col": "DATE", "decfloat_col": "DECFLOAT", "float_col": "FLOAT",
            "timestamp_col": "TIMESTAMP", "double_col": "DOUBLE", "bigint_col": "BIGINT",
            "smallint_col": "SMALLINT", "bit_col": "BIT", "obj_col": "OBJECT",
            "int_col": "INT", "bool_col": "BOOLEAN", "bytes_col": "BYTES",
            "interval_col": "INTERVAL", "array_col": "ARRAY<STRING>",
        }
    }
    merge_subqueries_fn = partial(
        optimizer.optimize,
        rules=[
            optimizer.qualify_tables.qualify_tables,
            optimizer.qualify_columns.qualify_columns,
            optimizer.merge_subqueries.merge_subqueries,
        ],
    )
    canonicalize_fn = partial(
        optimizer.optimize,
        rules=[
            optimizer.qualify.qualify,
            optimizer.qualify_columns.quote_identifiers,
            annotate_types,
            optimizer.canonicalize.canonicalize,
        ],
    )
    canon_internal_schema = {
        **base_schema,
        "jtbl": {"j": "JSON"},
        "pvt": {"c": "TEXT", "v": "INT"},
    }

    opt_fixtures_subcov = {
        "optimizer/annotate_functions.sql": _run_opt_fixture_cov(
            "optimizer/annotate_functions.sql",
            topt.annotate_functions,
            {"schema": annotate_funcs_schema},
            is_annotate_funcs=True,
        ),
        "optimizer/qualify_columns.sql": _run_opt_fixture_cov(
            "optimizer/qualify_columns.sql", topt.qualify_columns, {"schema": base_schema}
        ),
        "optimizer/qualify_columns_ddl.sql": _run_opt_fixture_cov(
            "optimizer/qualify_columns_ddl.sql", topt.qualify_columns, {"schema": base_schema}
        ),
        "optimizer/merge_subqueries.sql": _run_opt_fixture_cov(
            "optimizer/merge_subqueries.sql", merge_subqueries_fn, {"schema": base_schema}, execute_default=True
        ),
        "optimizer/canonicalize.sql": _run_opt_fixture_cov(
            "optimizer/canonicalize.sql", canonicalize_fn, {"schema": base_schema}
        ),
        "optimizer/canonicalize_internal_names.sql": _run_opt_fixture_cov(
            "optimizer/canonicalize_internal_names.sql",
            topt.qualify_then_canonicalize,
            {"schema": canon_internal_schema, "catalog": "c", "db": "db"},
        ),
        "optimizer/simplify.sql": _run_opt_fixture_cov(
            "optimizer/simplify.sql", topt.simplify, {"schema": base_schema}
        ),
        "optimizer/optimizer.sql": _run_opt_fixture_cov(
            "optimizer/optimizer.sql",
            optimizer.optimize,
            {"schema": {
                "x": {"a": "INT", "b": "INT"},
                "y": {"b": "INT", "c": "INT"},
                "z": {"a": "INT", "c": "INT"},
                "u": {"f": "INT", "g": "INT", "h": "TEXT"},
            }},
            pretty=True,
            execute_default=True,
        ),
        "optimizer/unnest_subqueries.sql": _run_opt_fixture_cov(
            "optimizer/unnest_subqueries.sql", optimizer.unnest_subqueries.unnest_subqueries, {}
        ),
        "optimizer/tpc-h/tpc-h.sql": _run_opt_fixture_cov(
            "optimizer/tpc-h/tpc-h.sql", optimizer.optimize, {"schema": TPCH_SCHEMA}, pretty=True
        ),
        "optimizer/eliminate_joins.sql": _run_opt_fixture_cov(
            "optimizer/eliminate_joins.sql", optimizer.eliminate_joins.eliminate_joins, {}, pretty=True
        ),
        "optimizer/pushdown_projections.sql": _run_opt_fixture_cov(
            "optimizer/pushdown_projections.sql", topt.pushdown_projections, {"schema": base_schema}
        ),
        "optimizer/pushdown_predicates.sql": _run_opt_fixture_cov(
            "optimizer/pushdown_predicates.sql", optimizer.pushdown_predicates.pushdown_predicates, {}
        ),
        "optimizer/annotate_types.sql": _run_opt_fixture_cov(
            "optimizer/annotate_types.sql", annotate_types, {}
        ),
    }

    return {
        "tpcds_exec_queries": tpcds_exec_queries,
        "tpcds_opt_cov": {k: dict(v) for k, v in tpcds_opt_cov.items()},
        "tpcds_exec_cov": {k: dict(v) for k, v in tpcds_exec_cov.items()},
        "tpcds_all_opt_cov": {k: dict(v) for k, v in tpcds_all_opt_cov.items()},
        "tpch_opt_hashes": tpch_opt_hashes,
        "gen_mem_dialects": gen_mem_dialects,
        "gen_mem_cov": {k: dict(v) for k, v in gen_mem_cov.items()},
        "scope_sec_cov": {k: dict(v) for k, v in scope_sec_cov.items()},
        "opt_fixtures": opt_fixtures_subcov,
    }


def compact_line_cov(line_cov: dict[str, dict[int, int]]) -> tuple[list[int], dict[str, list[tuple[int, int]]]]:
    pool: list[int] = []
    pool_idx: dict[int, int] = {}
    compact: dict[str, list[tuple[int, int]]] = {}
    for rel, lmap in line_cov.items():
        items = []
        for ln, mask in sorted(lmap.items()):
            idx = pool_idx.get(mask)
            if idx is None:
                idx = len(pool)
                pool_idx[mask] = idx
                pool.append(mask)
            items.append((ln, idx))
        compact[rel] = items
    return pool, compact


def main() -> None:
    t_start = time.perf_counter()
    tests = json.loads(TESTS_JSON.read_text("utf-8"))["tests"]
    test_files = sorted(set(t["file"] for t in tests))

    print(f"[build_maps] Building V3 maps for {len(tests)} tests across {len(test_files)} files...", file=sys.stderr)
    core_sqlglot_mods, eager_imports = collect_eager_imports(test_files)
    static_info = build_static_ast_info()
    dyn = run_dynamic_profiling(tests, core_sqlglot_mods)

    main_pool, main_cov = compact_line_cov(dyn["line_cov"])
    sub = dyn["subtests"]
    sub_packed = {
        "tpcds_exec_queries": sub["tpcds_exec_queries"],
        "tpcds_opt_cov": compact_line_cov(sub["tpcds_opt_cov"]),
        "tpcds_exec_cov": compact_line_cov(sub["tpcds_exec_cov"]),
        "tpcds_all_opt_cov": compact_line_cov(sub["tpcds_all_opt_cov"]),
        "tpch_opt_hashes": sub["tpch_opt_hashes"],
        "gen_mem_dialects": sub["gen_mem_dialects"],
        "gen_mem_cov": compact_line_cov(sub["gen_mem_cov"]),
        "scope_sec_cov": compact_line_cov(sub["scope_sec_cov"]),
        "opt_fixtures": {
            k: {
                "count": v["count"],
                "cov": compact_line_cov(v["cov"]),
                "dialects": v["dialects"],
            }
            for k, v in sub["opt_fixtures"].items()
        },
    }

    raw_lc = dyn["line_cov"]
    sym_ref_mask: dict[str, int] = {}
    for sym, refs in static_info["sym_ref_lines"].items():
        m = 0
        for rel, ln in refs:
            fl = raw_lc.get(rel)
            if fl:
                m |= fl.get(ln, 0)
        if m:
            sym_ref_mask[sym] = m

    payload = {
        "version": 3,
        "test_ids": dyn["test_ids"],
        "core_sqlglot_mods": core_sqlglot_mods,
        "eager_imports": eager_imports,
        "mod_to_rel": static_info["mod_to_rel"],
        "main_pool": main_pool,
        "main_cov": main_cov,
        "dialect_usage": dyn["dialect_usage"],
        "expr_usage": dyn["expr_usage"],
        "dict_key_cov": dyn["dict_key_cov"],
        "wrapped_dict_owners": dyn["wrapped_dict_owners"],
        "file_reads": dyn["file_reads"],
        "lazy_imports": dyn["lazy_imports"],
        "subclass_maps": dyn["subclass_maps"],
        "expr_subclasses": dyn["expr_subclasses"],
        "sym_ref_mask": sym_ref_mask,
        "subtests": sub_packed,
    }

    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    comp = zlib.compress(raw, level=9)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(comp)
    print(
        f"[build_maps] Done in {time.perf_counter() - t_start:.2f}s! "
        f"raw={len(raw)/1024:.1f}KB, compressed={len(comp)/1024:.1f}KB, "
        f"dict_owners={len(dyn['wrapped_dict_owners'])}, dict_entries={sum(len(v) for v in dyn['dict_key_cov'].values())}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
