"""Stage 1: Multi-layer build-time map loader and sound impact analyzer (V3)."""

from __future__ import annotations

import collections
import pickle
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from patch_utils import (
    FilePatch,
    IdentityOpEdit,
    analyze_python_patch,
    apply_file_patch,
    diff_sql_fixture_pairs,
    diff_sql_fixtures_single,
    parse_patch,
)

SINGLE_SQL_FIXTURES = {
    "tests/fixtures/identity.sql",
    "tests/fixtures/partial.sql",
    "tests/fixtures/optimizer/qualify_columns__invalid.sql",
}

FIXTURE_TO_OPT_TEST_ID = {
    "optimizer/optimizer.sql": "tests/test_optimizer.py::TestOptimizer::test_optimize",
    "optimizer/isolate_table_selects.sql": "tests/test_optimizer.py::TestOptimizer::test_isolate_table_selects",
    "optimizer/qualify_tables.sql": "tests/test_optimizer.py::TestOptimizer::test_qualify_tables",
    "optimizer/normalize.sql": "tests/test_optimizer.py::TestOptimizer::test_normalize",
    "optimizer/qualify_columns.sql": "tests/test_optimizer.py::TestOptimizer::test_qualify_columns",
    "optimizer/qualify_columns_ddl.sql": "tests/test_optimizer.py::TestOptimizer::test_qualify_columns",
    "optimizer/qualify_columns__with_invisible.sql": "tests/test_optimizer.py::TestOptimizer::test_qualify_columns__with_invisible",
    "optimizer/pushdown_cte_alias_columns.sql": "tests/test_optimizer.py::TestOptimizer::test_pushdown_cte_alias_columns",
    "optimizer/qualify_columns__invalid.sql": "tests/test_optimizer.py::TestOptimizer::test_qualify_columns__invalid",
    "optimizer/normalize_identifiers.sql": "tests/test_optimizer.py::TestOptimizer::test_normalize_identifiers",
    "optimizer/quote_identifiers.sql": "tests/test_optimizer.py::TestOptimizer::test_quote_identifiers",
    "optimizer/pushdown_projections.sql": "tests/test_optimizer.py::TestOptimizer::test_pushdown_projection",
    "optimizer/simplify.sql": "tests/test_optimizer.py::TestOptimizer::test_simplify",
    "optimizer/unnest_subqueries.sql": "tests/test_optimizer.py::TestOptimizer::test_unnest_subqueries",
    "optimizer/pushdown_predicates.sql": "tests/test_optimizer.py::TestOptimizer::test_pushdown_predicates",
    "optimizer/optimize_joins.sql": "tests/test_optimizer.py::TestOptimizer::test_optimize_joins",
    "optimizer/eliminate_joins.sql": "tests/test_optimizer.py::TestOptimizer::test_eliminate_joins",
    "optimizer/eliminate_ctes.sql": "tests/test_optimizer.py::TestOptimizer::test_eliminate_ctes",
    "optimizer/merge_subqueries.sql": "tests/test_optimizer.py::TestOptimizer::test_merge_subqueries",
    "optimizer/eliminate_subqueries.sql": "tests/test_optimizer.py::TestOptimizer::test_eliminate_subqueries",
    "optimizer/canonicalize_internal_names.sql": "tests/test_optimizer.py::TestOptimizer::test_canonicalize_internal_names",
    "optimizer/canonicalize.sql": "tests/test_optimizer.py::TestOptimizer::test_canonicalize",
    "optimizer/tpc-h/tpc-h.sql": "tests/test_optimizer.py::TestOptimizer::test_tpch",
    "optimizer/tpc-ds/tpc-ds.sql": "tests/test_optimizer.py::TestOptimizer::test_tpcds",
    "optimizer/annotate_types.sql": "tests/test_optimizer.py::TestOptimizer::test_annotate_types",
    "optimizer/annotate_functions.sql": "tests/test_optimizer.py::TestOptimizer::test_annotate_funcs",
}

# Class attributes processed by _Dialect/_Tokenizer/_Parser/_Generator metaclasses at import time
# (MUST NOT be intersected with runtime sym_ref_mask!)
METACLASS_PROCESSED_ATTRS = frozenset({
    "KEYWORDS", "SINGLE_TOKENS", "QUOTES", "IDENTIFIERS", "COMMENTS", "STRING_ESCAPES",
    "BIT_STRINGS", "HEX_STRINGS", "BYTE_STRINGS", "RAW_STRINGS", "HEREDOC_STRINGS",
    "UNICODE_STRINGS", "VAR_SINGLE_TOKENS", "ESCAPED_SEQUENCES", "NUMERIC_ESCAPES",
    "TIME_MAPPING", "FORMAT_MAPPING", "INVERSE_TIME_MAPPING", "TIME_TRIE",
    "TRANSFORMS", "RESERVED_TOKENS", "SHOW_PARSERS", "SHOW_TRIE",
    "UNSUPPORTED_MESSAGES", "NORMALIZATION_STRATEGY", "DATE_PART_MAPPING",
    "VALID_INTERVAL_UNITS", "SUPPORTS_COLUMN_JOIN_MARKS", "EXPRESSION_METADATA",
    "FUNCTIONS", "STATEMENT_PARSERS", "FUNCTION_PARSERS", "PROPERTY_PARSERS",
    "TYPE_MAPPING", "PROPERTIES_LOCATION", "arg_types",
})


def _expand_compact_cov(packed: tuple[list[int], dict[str, list[tuple[int, int]]]]) -> dict[str, dict[int, int]]:
    pool, compact = packed
    out: dict[str, dict[int, int]] = {}
    for rel, pairs in compact.items():
        out[rel] = {ln: pool[idx] for ln, idx in pairs}
    return out


class ImpactMaps:
    def __init__(self, map_path: Path, runtime_tests: list[dict]):
        data = pickle.loads(zlib.decompress(map_path.read_bytes()))
        self.map_test_ids: list[str] = data["test_ids"]
        self.id_to_bit: dict[str, int] = {tid: i for i, tid in enumerate(self.map_test_ids)}
        self.core_sqlglot_mods: set[str] = set(data["core_sqlglot_mods"])
        self.eager_imports: dict[str, list[str]] = data["eager_imports"]
        self.mod_to_rel: dict[str, str] = data["mod_to_rel"]
        self.rel_to_mod: dict[str, str] = {v: k for k, v in self.mod_to_rel.items()}

        self.line_cov: dict[str, dict[int, int]] = _expand_compact_cov((data["main_pool"], data["main_cov"]))
        self.dialect_usage: dict[str, int] = data["dialect_usage"]
        self.expr_usage: dict[str, int] = data["expr_usage"]
        self.dict_key_cov: dict[tuple[str, str], dict[str, int]] = data.get("dict_key_cov", {})
        self.wrapped_dict_owners: set[tuple[str, str]] = set(data.get("wrapped_dict_owners", ()))
        self.file_reads: dict[str, int] = data["file_reads"]
        self.lazy_imports: dict[str, int] = data["lazy_imports"]
        self.subclass_maps: dict[str, list[str]] = data["subclass_maps"]
        self.expr_subclasses: dict[str, list[str]] = data["expr_subclasses"]
        self.sym_ref_mask: dict[str, int] = data["sym_ref_mask"]

        sub = data["subtests"]
        self.tpcds_exec_queries: list[dict] = sub["tpcds_exec_queries"]
        self.tpcds_opt_cov = _expand_compact_cov(sub["tpcds_opt_cov"])
        self.tpcds_exec_cov = _expand_compact_cov(sub["tpcds_exec_cov"])
        self.tpcds_all_opt_cov = _expand_compact_cov(sub["tpcds_all_opt_cov"])
        self.tpch_opt_hashes: list[tuple[str, str]] = sub["tpch_opt_hashes"]
        self.gen_mem_dialects: list[str] = sub["gen_mem_dialects"]
        self.gen_mem_cov = _expand_compact_cov(sub["gen_mem_cov"])
        self.scope_sec_cov = _expand_compact_cov(sub["scope_sec_cov"]) if "scope_sec_cov" in sub else {}
        self.opt_fixtures: dict[str, dict] = {}
        for fk, fv in sub["opt_fixtures"].items():
            self.opt_fixtures[fk] = {
                "count": fv["count"],
                "cov": _expand_compact_cov(fv["cov"]),
                "dialects": fv["dialects"],
            }

        self.runtime_tests = runtime_tests
        self.bit_to_num: list[int | None] = [None] * len(self.map_test_ids)
        self.num_to_test: dict[int, dict] = {}
        self.id_to_num: dict[str, int] = {}
        self.file_to_nums: dict[str, list[int]] = collections.defaultdict(list)
        self.file_to_mask: dict[str, int] = collections.defaultdict(int)
        self.unmapped_nums: list[int] = []

        for t in runtime_tests:
            num = t["number"]
            tid = t["id"]
            tf = t["file"]
            self.num_to_test[num] = t
            self.id_to_num[tid] = num
            self.file_to_nums[tf].append(num)
            b = self.id_to_bit.get(tid)
            if b is not None:
                self.bit_to_num[b] = num
                self.file_to_mask[tf] |= (1 << b)
            else:
                self.unmapped_nums.append(num)

        self.rev_eager_mask: dict[str, int] = collections.defaultdict(int)
        for tf, mods in self.eager_imports.items():
            fmask = self.file_to_mask.get(tf, 0)
            if not fmask:
                continue
            self.rev_eager_mask[tf] |= fmask
            for rel in mods:
                self.rev_eager_mask[rel] |= fmask

        self.file_union_cov: dict[str, int] = {}
        for rel, lmap in self.line_cov.items():
            u = 0
            for m in lmap.values():
                u |= m
            self.file_union_cov[rel] = u

        # Precompute exact set of repo files imported by tests/dialects/test_dialect.py & tests/test_optimizer.py
        self.test_dialect_eager_set: set[str] = set(
            self.eager_imports.get("tests/dialects/test_dialect.py", ())
        )
        self.test_dialect_eager_set.add("tests/dialects/test_dialect.py")
        self.test_optimizer_eager_set: set[str] = set(
            self.eager_imports.get("tests/test_optimizer.py", ())
        )
        self.test_optimizer_eager_set.add("tests/test_optimizer.py")
        self.all_mask = (1 << len(self.map_test_ids)) - 1

    def mask_to_nums(self, mask: int) -> list[int]:
        nums: list[int] = []
        m = mask
        b2n = self.bit_to_num
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            num = b2n[idx]
            if num is not None:
                nums.append(num)
            m ^= lsb
        return nums

    def mask_for_dialect_class(self, mod_name: str, cls_qual: str) -> int:
        key = f"{mod_name}:{cls_qual}"
        mask = self.dialect_usage.get(key, 0)
        for sub_key in self.subclass_maps.get(key, ()):
            mask |= self.dialect_usage.get(sub_key, 0)
        if not mask:
            mask |= self.dialect_usage.get(mod_name, 0)
        return mask

    def mask_for_expr_class(self, cls_name: str) -> int:
        mask = self.expr_usage.get(cls_name, 0)
        for sub_name in self.expr_subclasses.get(cls_name, ()):
            mask |= self.expr_usage.get(sub_name, 0)
        return mask

    def lookup_tracked_dict_keys(self, mod_name: str, cls_qual: str, attr: str, keys: tuple[str, ...]) -> tuple[bool, int]:
        """If (mod:cls_qual, attr) (and all its subclasses defining attr) was wrapped with TrackedDict,
        return (True, bitmask_of_tests_accessing_any_key_or_wildcard).
        Otherwise return (False, 0)."""
        owner = f"{mod_name}:{cls_qual}"
        if (owner, attr) not in self.wrapped_dict_owners:
            return False, 0
        owners_to_check = [owner] + [s for s in self.subclass_maps.get(owner, ()) if (s, attr) in self.wrapped_dict_owners]
        mask = 0
        for own in owners_to_check:
            kmap = self.dict_key_cov.get((own, attr))
            if kmap:
                mask |= kmap.get("*", 0)
                for k in keys:
                    mask |= kmap.get(k, 0)
        return True, mask


@dataclass
class CommitAnalysis:
    commit_id: str
    file_patches: list[FilePatch]
    patched_files: dict[str, str | None]
    is_inert: bool = False
    apply_failed: bool = False
    syntax_error_files: list[str] = field(default_factory=list)
    candidate_nums: list[int] = field(default_factory=list)
    needs_clean_sqlglot: bool = False
    touches_executor_or_planner: bool = False
    touches_serde: bool = False
    touches_duckdb: bool = False
    touches_any_py: bool = False
    py_changed_lines: dict[str, set[int]] = field(default_factory=dict)
    py_class_or_top_files: set[str] = field(default_factory=set)
    py_dialect_mods: set[str] = field(default_factory=set)
    fixture_changed_indices: dict[str, list[int]] = field(default_factory=dict)
    confirmed_fail_nums: set[int] = field(default_factory=set)
    # If ALL non-inert edits in this commit are pure identity-op swaps (`is`<->`==` / `is not`<->`!=`)
    pure_identity_edits: list[IdentityOpEdit] = field(default_factory=list)
    all_edits_are_identity_op: bool = False
    # If ALL non-inert edits are confined to specific non-base dialect modules (plus subclasses)
    only_dialect_stems: set[str] | None = None
    # True if commit touches any module imported by tests.dialects.test_dialect (BigQuery/Hive/Snowflake/Spark2)
    touches_test_dialect_deps: bool = False
    # True if commit touches any module imported by tests.test_optimizer
    touches_test_optimizer_deps: bool = False


TEST_DIALECT_PRELOAD_STEMS = frozenset({
    "bigquery", "hive", "snowflake", "spark2", "spark", "databricks", "athena", "duckdb",
})


def _dialect_stem_from_path(rel: str) -> str | None:
    if rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
        stem = Path(rel).stem
        if stem not in ("__init__", "dialect", "base"):
            return stem
    return None


def analyze_commit(commit_id: str, patch_text: str, sqlglot_root: Path, maps: ImpactMaps) -> CommitAnalysis:
    fps = parse_patch(patch_text)
    analysis = CommitAnalysis(commit_id=commit_id, file_patches=fps, patched_files={})

    if not fps:
        analysis.is_inert = True
        return analysis

    cand_mask = 0
    all_inert = True
    all_identity = True
    all_leaf_dialects = True
    leaf_dialect_stems: set[str] = set()

    for fp in fps:
        rel = fp.path
        orig_rel = fp.old_path or rel
        orig_path = sqlglot_root / orig_rel
        orig_text: str | None = None
        if not fp.is_new and orig_path.is_file():
            try:
                orig_text = orig_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                orig_text = None

        ok, new_text = apply_file_patch(orig_text, fp)
        if not ok:
            analysis.apply_failed = True
            analysis.is_inert = False
            analysis.candidate_nums = [t["number"] for t in maps.runtime_tests]
            return analysis

        def _record_patched() -> None:
            if fp.is_rename and fp.old_path and fp.new_path and fp.old_path != fp.new_path:
                analysis.patched_files[fp.old_path] = None
                analysis.patched_files[fp.new_path] = new_text
            else:
                analysis.patched_files[rel] = new_text

        if rel.endswith(".py"):
            is_test_file = rel.startswith("tests/") and (
                rel.startswith("tests/test_") or rel.startswith("tests/dialects/test_")
            )
            pdiff = analyze_python_patch(orig_text, new_text, fp, is_test_file=is_test_file)

            if pdiff.is_inert:
                continue

            _record_patched()
            all_inert = False
            analysis.touches_any_py = True

            if pdiff.identity_op_edit is not None:
                analysis.pure_identity_edits.append(pdiff.identity_op_edit)
            else:
                all_identity = False

            if rel in maps.test_dialect_eager_set:
                analysis.touches_test_dialect_deps = True
            if rel in maps.test_optimizer_eager_set:
                analysis.touches_test_optimizer_deps = True
            dstem = _dialect_stem_from_path(rel)
            if dstem is not None:
                leaf_dialect_stems.add(dstem)
                # Expand with all subclasses of this dialect/parser/generator
                for prefix in ("sqlglot.dialects.", "sqlglot.parsers.", "sqlglot.generators."):
                    m_prefix = prefix + dstem + ":"
                    for ck, subs in maps.subclass_maps.items():
                        if ck.startswith(m_prefix):
                            for sk in subs:
                                sub_mod = sk.split(":")[0]
                                sub_stem = sub_mod.split(".")[-1]
                                if sub_stem not in ("dialect", "base", "__init__"):
                                    leaf_dialect_stems.add(sub_stem)
                if dstem in TEST_DIALECT_PRELOAD_STEMS or (leaf_dialect_stems & TEST_DIALECT_PRELOAD_STEMS):
                    analysis.touches_test_dialect_deps = True
            else:
                all_leaf_dialects = False

            if rel in maps.core_sqlglot_mods:
                analysis.needs_clean_sqlglot = True
            if (
                rel.startswith("sqlglot/executor/")
                or rel == "sqlglot/planner.py"
                or rel == "sqlglot/generators/python.py"
            ):
                analysis.touches_executor_or_planner = True
            if rel == "sqlglot/serde.py" or rel == "sqlglot/expressions/core.py":
                analysis.touches_serde = True
            if (
                "duckdb" in rel
                or rel in maps.core_sqlglot_mods
                or rel == "sqlglot/dialects/dialect.py"
            ):
                analysis.touches_duckdb = True

            if pdiff.syntax_error:
                analysis.syntax_error_files.append(rel)
                eager_m = maps.rev_eager_mask.get(rel, 0)
                lazy_m = maps.lazy_imports.get(rel, 0)
                file_m = maps.file_reads.get(rel, 0) | maps.file_union_cov.get(rel, 0)
                mod_name = maps.rel_to_mod.get(rel, "")
                # Include EVERY test using any dialect/parser/generator/tokenizer defined in this module
                # (catches lazy-imported dialects like Dremio/Postgres when syntax error breaks import!)
                dial_m = maps.dialect_usage.get(mod_name, 0) if mod_name else 0
                if mod_name:
                    m_pref = mod_name + ":"
                    for ck, subs in maps.subclass_maps.items():
                        if ck.startswith(m_pref):
                            dial_m |= maps.dialect_usage.get(ck, 0)
                            for sk in subs:
                                dial_m |= maps.dialect_usage.get(sk, 0)
                if is_test_file:
                    file_m |= maps.file_to_mask.get(rel, 0)
                for num in maps.mask_to_nums(eager_m | dial_m):
                    analysis.confirmed_fail_nums.add(num)
                cand_mask |= eager_m | lazy_m | file_m | dial_m
                analysis.py_class_or_top_files.add(rel)
                continue

            cand_mask |= maps.file_reads.get(rel, 0)

            if is_test_file and pdiff.only_test_methods is not None:
                for cname, mname in pdiff.only_test_methods:
                    tid = f"{rel}::{cname}::{mname}"
                    b = maps.id_to_bit.get(tid)
                    if b is not None:
                        cand_mask |= (1 << b)
                    else:
                        cand_mask |= maps.file_to_mask.get(rel, 0)
                fl = maps.line_cov.get(rel, {})
                for ln in pdiff.func_lines:
                    cand_mask |= fl.get(ln, 0)
                analysis.py_changed_lines.setdefault(rel, set()).update(pdiff.func_lines)
                continue

            if pdiff.func_lines:
                fl = maps.line_cov.get(rel, {})
                for ln in pdiff.func_lines:
                    cand_mask |= fl.get(ln, 0)
                analysis.py_changed_lines.setdefault(rel, set()).update(pdiff.func_lines)

            if pdiff.class_body_edits:
                analysis.py_class_or_top_files.add(rel)
                mod_name = maps.rel_to_mod.get(rel, "")
                for cbe in pdiff.class_body_edits:
                    cls_qual = cbe.cls_qual
                    attr_name = cbe.attr_name
                    top_cls = cls_qual.split(".")[0]
                    base_cls = cls_qual.split(".")[-1]

                    # Fast exact TrackedDict key-level lookup!
                    if attr_name and cbe.dict_value_keys:
                        ok_td, td_mask = maps.lookup_tracked_dict_keys(
                            mod_name, cls_qual, attr_name, cbe.dict_value_keys
                        )
                        if ok_td:
                            cand_mask |= td_mask
                            continue

                    hit = 0
                    if rel.startswith("sqlglot/expressions/"):
                        hit |= maps.mask_for_expr_class(base_cls)
                        hit |= maps.sym_ref_mask.get(base_cls, 0)
                        if top_cls != base_cls:
                            hit |= maps.mask_for_expr_class(top_cls)
                            hit |= maps.sym_ref_mask.get(top_cls, 0)
                    elif (
                        rel.startswith("sqlglot/dialects/")
                        or rel.startswith("sqlglot/parsers/")
                        or rel.startswith("sqlglot/generators/")
                        or rel in ("sqlglot/parser.py", "sqlglot/generator.py", "sqlglot/tokens.py", "sqlglot/tokenizer_core.py")
                    ):
                        dial_hit = maps.mask_for_dialect_class(mod_name, cls_qual)
                        dial_hit |= maps.mask_for_dialect_class(mod_name, top_cls)
                        dial_hit |= maps.sym_ref_mask.get(top_cls, 0)
                        if base_cls not in ("Tokenizer", "Parser", "Generator"):
                            dial_hit |= maps.sym_ref_mask.get(base_cls, 0)

                        # Precision intersection for non-metaclass scalar class attributes (e.g. TSQL.WEEK_OFFSET)!
                        if (
                            attr_name
                            and not cbe.is_dict_assign
                            and attr_name not in METACLASS_PROCESSED_ATTRS
                            and attr_name in maps.sym_ref_mask
                            and dial_hit
                        ):
                            hit = dial_hit & maps.sym_ref_mask[attr_name]
                            # Also keep tests referencing the class + attr directly
                            if not hit:
                                hit = dial_hit
                        else:
                            hit = dial_hit

                        analysis.py_dialect_mods.add(mod_name)
                        for sk in maps.subclass_maps.get(f"{mod_name}:{top_cls}", ()):
                            analysis.py_dialect_mods.add(sk.split(":")[0])
                        for sk in maps.subclass_maps.get(f"{mod_name}:{cls_qual}", ()):
                            analysis.py_dialect_mods.add(sk.split(":")[0])
                    else:
                        hit |= maps.sym_ref_mask.get(base_cls, 0)
                        hit |= maps.file_union_cov.get(rel, 0)
                    if not hit:
                        hit |= maps.file_union_cov.get(rel, 0) | maps.lazy_imports.get(rel, 0)
                        if not hit:
                            hit |= maps.rev_eager_mask.get(rel, 0)
                    cand_mask |= hit

            if pdiff.module_level_symbols:
                analysis.py_class_or_top_files.add(rel)
                for sym in pdiff.module_level_symbols:
                    hit = maps.sym_ref_mask.get(sym, 0)
                    if rel.startswith("sqlglot/typing/"):
                        d_stem = Path(rel).stem
                        if d_stem in ("__init__", "dialect"):
                            hit |= maps.file_union_cov.get("sqlglot/optimizer/annotate_types.py", 0)
                        else:
                            d_mod = f"sqlglot.dialects.{d_stem}"
                            hit = maps.dialect_usage.get(d_mod, 0) | maps.file_reads.get(rel, 0)
                            for ck, subs in maps.subclass_maps.items():
                                if ck.startswith(d_mod + ":"):
                                    for sk in subs:
                                        hit |= maps.dialect_usage.get(sk, 0)
                    if not hit:
                        hit |= maps.file_union_cov.get(rel, 0) | maps.lazy_imports.get(rel, 0)
                    cand_mask |= hit

            if pdiff.has_import_edit or pdiff.has_unknown_top_edit or fp.is_new or fp.is_deleted:
                analysis.py_class_or_top_files.add(rel)
                if is_test_file:
                    cand_mask |= maps.file_to_mask.get(rel, 0)
                else:
                    mod_name = maps.rel_to_mod.get(rel, "")
                    dial_m = maps.dialect_usage.get(mod_name, 0) if mod_name else 0
                    if mod_name:
                        m_pref = mod_name + ":"
                        for ck, subs in maps.subclass_maps.items():
                            if ck.startswith(m_pref):
                                dial_m |= maps.dialect_usage.get(ck, 0)
                                for sk in subs:
                                    dial_m |= maps.dialect_usage.get(sk, 0)
                    hit = (
                        maps.file_union_cov.get(rel, 0)
                        | maps.lazy_imports.get(rel, 0)
                        | maps.rev_eager_mask.get(rel, 0)
                        | dial_m
                    )
                    cand_mask |= hit if hit else maps.all_mask

            continue

        all_identity = False
        all_leaf_dialects = False

        if rel.startswith("tests/fixtures/"):
            sub_fix = rel[len("tests/fixtures/"):]
            if rel.endswith(".sql") and orig_text is not None and new_text is not None:
                if rel in SINGLE_SQL_FIXTURES:
                    inert, ch_idx = diff_sql_fixtures_single(orig_text, new_text)
                else:
                    inert, ch_idx = diff_sql_fixture_pairs(orig_text, new_text)
                if inert:
                    continue
                _record_patched()
                all_inert = False
                analysis.fixture_changed_indices[sub_fix] = ch_idx
                f_mask = maps.file_reads.get(rel, 0)
                if sub_fix == "optimizer/tpc-ds/tpc-ds.sql":
                    for tid in (
                        "tests/test_optimizer.py::TestOptimizer::test_tpcds",
                        "tests/test_executor.py::TestExecutor::test_execute_tpcds",
                    ):
                        b = maps.id_to_bit.get(tid)
                        if b is not None:
                            cand_mask |= (1 << b)
                elif sub_fix == "optimizer/tpc-h/tpc-h.sql":
                    for tid in (
                        "tests/test_optimizer.py::TestOptimizer::test_tpch",
                        "tests/test_executor.py::TestExecutor::test_execute_tpch",
                        "tests/test_executor.py::TestExecutor::test_optimized_tpch",
                    ):
                        b = maps.id_to_bit.get(tid)
                        if b is not None:
                            cand_mask |= (1 << b)
                else:
                    cand_mask |= f_mask
                    opt_tid = FIXTURE_TO_OPT_TEST_ID.get(sub_fix)
                    if opt_tid and opt_tid in maps.id_to_bit:
                        cand_mask |= (1 << maps.id_to_bit[opt_tid])
            else:
                _record_patched()
                all_inert = False
                if "tpc-ds" in rel or "tpc-h" in rel:
                    analysis.touches_executor_or_planner = True
                    analysis.touches_duckdb = True
                cand_mask |= maps.file_reads.get(rel, 0) or maps.all_mask
            continue

        f_mask = maps.file_reads.get(rel, 0)
        if f_mask:
            _record_patched()
            all_inert = False
            cand_mask |= f_mask
        elif rel in ("pyproject.toml", "setup.py", "setup.cfg") or rel.startswith("sqlglotc/"):
            _record_patched()
            all_inert = False
            for tid in (
                "tests/test_sqlglotc.py::TestBuildExt::test_build_removes_obsolete_extensions",
                "tests/test_sqlglotc.py::TestBuildExt::test_build_removes_stale_directories",
                "tests/test_sqlglotc.py::TestBuildExt::test_inplace_removes_obsolete_extensions",
                "tests/test_dialect_entry_points.py::TestDialectEntryPoints::test_entry_point_plugin_discovery_modern_api",
                "tests/test_dialect_entry_points.py::TestDialectEntryPoints::test_entry_point_plugin_discovery_legacy_api",
                "tests/test_dialect_entry_points.py::TestDialectEntryPoints::test_entry_point_plugin_not_found",
            ):
                b = maps.id_to_bit.get(tid)
                if b is not None:
                    cand_mask |= (1 << b)

    if all_inert:
        analysis.is_inert = True
        analysis.candidate_nums = []
        return analysis

    analysis.all_edits_are_identity_op = bool(all_identity and analysis.pure_identity_edits)
    if all_leaf_dialects and leaf_dialect_stems:
        analysis.only_dialect_stems = leaf_dialect_stems
    nums = maps.mask_to_nums(cand_mask)
    if maps.unmapped_nums:
        nums.extend(maps.unmapped_nums)
    analysis.candidate_nums = sorted(set(nums))
    return analysis
