#!/usr/bin/env python3
"""Pre-pickle TPC-H and TPC-DS Table objects during Docker build so runtime
loads them in ~15ms instead of ~500ms of gzip+CSV+literal_eval parsing."""

from __future__ import annotations

import ast
import csv
import gzip
import os
import pickle
import sys
from pathlib import Path

os.environ["SKIP_INTEGRATION"] = "0"
sys.path.insert(0, "/app/sqlglot")

from sqlglot.executor.table import Table
from tests.helpers import FIXTURES_DIR, TPCH_SCHEMA, TPCDS_SCHEMA


def load_tables(directory: str, schema: dict) -> dict[str, Table]:
    tables: dict[str, Table] = {}
    for table, columns in schema.items():
        fn = f"{directory}{table}.csv.gz"
        with gzip.open(fn, "rt", newline="") as f:
            reader = csv.reader(f, delimiter="|")
            next(reader)
            rows = []
            ctypes = []
            for row in reader:
                if not ctypes:
                    for v in row:
                        try:
                            ctypes.append(type(ast.literal_eval(v)))
                        except (ValueError, SyntaxError):
                            ctypes.append(str)
                rows.append(
                    tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row))
                )
        tables[table] = Table(columns=columns, rows=rows)
    return tables


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/maps/tables.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpch = load_tables(FIXTURES_DIR + "/optimizer/tpc-h/", TPCH_SCHEMA)
    tpcds = load_tables(FIXTURES_DIR + "/optimizer/tpc-ds/", TPCDS_SCHEMA)
    with open(out_path, "wb") as f:
        pickle.dump({"tpch": tpch, "tpcds": tpcds}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[build_tables_cache] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
