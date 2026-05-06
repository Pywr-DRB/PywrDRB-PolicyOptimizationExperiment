#!/usr/bin/env python3
"""
Rename legacy Borg output files from ``*_mrfmasked_*`` to ``*_mrffiltered_*``.

By default only renames under ``outputs/`` (CSV, .set, .info, .runtime as matched).
Use --dry-run to list planned renames without touching files.

Example (from CEE6400Project/):

  python -m methods.analysis.migrate_mrfmasked_outputs_to_mrffiltered --dry-run
  python -m methods.analysis.migrate_mrfmasked_outputs_to_mrffiltered
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def planned_pairs(root: Path, exts: tuple[str, ...]) -> list[tuple[Path, Path]]:
    out: list[tuple[Path, Path]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if "_mrfmasked_" not in name:
                continue
            if exts and not any(name.endswith(e) for e in exts):
                continue
            new_name = name.replace("_mrfmasked_", "_mrffiltered_", 1)
            if new_name == name:
                continue
            old_p = Path(dirpath) / name
            new_p = Path(dirpath) / new_name
            if new_p.exists():
                continue
            out.append((old_p, new_p))
    return sorted(out, key=lambda t: str(t[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs",
        help="Directory tree to scan (default: project outputs/)",
    )
    ap.add_argument(
        "--ext",
        action="append",
        default=[".csv", ".set", ".info", ".runtime"],
        help="File suffixes to include (repeatable; default: csv set info runtime)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print renames only.")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"[error] not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    exts = tuple(args.ext)
    pairs = planned_pairs(root, exts)
    if not pairs:
        print(f"[info] no _mrfmasked_* files to rename under {root}")
        return

    for old_p, new_p in pairs:
        print(f"{old_p.relative_to(root)} -> {new_p.name}")

    if args.dry_run:
        print(f"[dry-run] {len(pairs)} file(s); run without --dry-run to apply.")
        return

    for old_p, new_p in pairs:
        old_p.rename(new_p)
    print(f"[done] renamed {len(pairs)} file(s).")


if __name__ == "__main__":
    main()
