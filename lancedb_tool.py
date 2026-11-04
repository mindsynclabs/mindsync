#!/usr/bin/env python3
"""Small CLI to inspect and clean a local LanceDB database.

Usage:
  python lancedb_tool.py list
  python lancedb_tool.py info knowledge_base
  python lancedb_tool.py view knowledge_base --limit 5
  python lancedb_tool.py view knowledge_base --doc-id my-document --full
  python lancedb_tool.py view knowledge_base --id my-document-chunk-0 --full
  python lancedb_tool.py drop knowledge_base
  python lancedb_tool.py drop-all --yes

Pick the database with --uri or the LANCEDB_URI env var (default: data/lancedb).
"""
import argparse
import os
import sys

import lancedb


def table_names(db):
    # list_tables() may return a list or a response object depending on the version
    res = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    return list(getattr(res, "tables", res))


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def cmd_list(db, args):
    names = table_names(db)
    if not names:
        print("No tables.")
        return
    for name in names:
        print(f"{name:30} rows: {db.open_table(name).count_rows()}")


def cmd_info(db, args):
    table = db.open_table(args.table)
    print(f"Table: {args.table}")
    print(f"Rows:  {table.count_rows()}")
    print("Schema:")
    print(table.schema)


def cmd_view(db, args):
    table = db.open_table(args.table)

    filters = []
    if args.id:
        filters.append(f"id = {sql_quote(args.id)}")
    if args.doc_id:
        filters.append(f"doc_id = {sql_quote(args.doc_id)}")

    query = table.search()
    if filters:
        query = query.where(" AND ".join(filters))
    rows = query.limit(args.limit).to_list()

    if not rows:
        print("No rows found.")
        return

    print(f"Showing {len(rows)} of {table.count_rows()} rows\n")
    for i, row in enumerate(rows, 1):
        print(f"--- row {i} " + "-" * 50)
        for key, value in row.items():
            if key == "vector":
                dim = len(value) if value is not None else 0
                print(f"{key}: <{dim} dims> {list(value[:3])}...")
            elif key.startswith("_"):
                continue  # internal columns such as _distance
            else:
                text = str(value)
                if not args.full and len(text) > args.width:
                    text = text[: args.width] + f"... [+{len(text) - args.width} chars]"
                print(f"{key}: {text}")
        print()


def confirm(message: str, yes: bool) -> bool:
    if yes:
        return True
    return input(f"{message} [y/N] ").strip().lower() == "y"


def cmd_drop(db, args):
    if args.table not in table_names(db):
        print(f"Table '{args.table}' not found. Existing: {table_names(db)}")
        sys.exit(1)
    if not confirm(f"Drop table '{args.table}'?", args.yes):
        print("Cancelled.")
        return
    db.drop_table(args.table)
    print(f"Dropped '{args.table}'. Remaining: {table_names(db)}")


def cmd_drop_all(db, args):
    names = table_names(db)
    if not names:
        print("No tables to drop.")
        return
    if not confirm(f"Drop ALL tables {names}?", args.yes):
        print("Cancelled.")
        return
    for name in names:
        db.drop_table(name)
        print(f"Dropped '{name}'")
    print(f"Remaining: {table_names(db)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect and clean a LanceDB database")
    parser.add_argument("--uri", default=os.getenv("LANCEDB_URI", "data/lancedb"))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list tables with row counts").set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="show schema and row count")
    p.add_argument("table")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("view", help="show rows (vectors are summarized)")
    p.add_argument("table")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--id", help="filter by chunk id")
    p.add_argument("--doc-id", help="filter by doc_id (needs a doc_id column)")
    p.add_argument("--full", action="store_true", help="do not truncate long text")
    p.add_argument("--width", type=int, default=200, help="max chars per field")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("drop", help="drop one table")
    p.add_argument("table")
    p.add_argument("--yes", action="store_true", help="skip confirmation")
    p.set_defaults(func=cmd_drop)

    p = sub.add_parser("drop-all", help="drop every table")
    p.add_argument("--yes", action="store_true", help="skip confirmation")
    p.set_defaults(func=cmd_drop_all)

    args = parser.parse_args()
    print(f"[db] {os.path.abspath(args.uri)}\n")
    db = lancedb.connect(args.uri)
    args.func(db, args)


if __name__ == "__main__":
    main()