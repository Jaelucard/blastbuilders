#!/usr/bin/env python3
"""Pre-pickle TPC-H and TPC-DS Table objects individually during Docker build so
runtime loads ONLY the specific tables a query references (~5ms instead of ~200ms)."""

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


def save_tables(directory: str, schema: dict, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
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
        t_obj = Table(columns=columns, rows=rows)
        dst = out_dir / f"{table}.pkl"
        data = pickle.dumps(t_obj, protocol=pickle.HIGHEST_PROTOCOL)
        dst.write_bytes(data)
        total += len(data)
    return total


def main() -> None:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/maps/tables")
    b1 = save_tables(FIXTURES_DIR + "/optimizer/tpc-h/", TPCH_SCHEMA, out_root / "tpch")
    b2 = save_tables(FIXTURES_DIR + "/optimizer/tpc-ds/", TPCDS_SCHEMA, out_root / "tpcds")
    print(f"[build_tables_cache] wrote per-table pickles to {out_root} ({(b1+b2)/1024:.1f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
