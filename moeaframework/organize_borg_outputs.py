#!/usr/bin/env python3
"""
Copy Borg island *.runtime files from CEE6400Project/outputs/ into the layout expected by
the MOEAFramework scripts:

  moeaframework/outputs/Policy_<POLICY>/runtime/<reservoir>/*.runtime

Also installs templates/1-header-file_<POLICY>.txt as Policy_<POLICY>/1-header-file.txt
when missing.

Filename pattern (multi-island / MMBorg):

  MMBorg_<n>M_<POLICY>_<RESERVOIR>_nfe<N>_seed<S>_<island>.runtime
  MMBorg_..._seed<S>_mrffiltered_regression_<island>.runtime
  MMBorg_..._seed<S>_mrffiltered_perfect_<island>.runtime
  MMBorg_..._seed<S>_mrfmasked_<island>.runtime (legacy)
  MMBorg_..._seed<S>_mrfmasked_perfect_<island>.runtime (legacy)

Run from the moeaframework/ directory:

  python organize_borg_outputs.py
  python organize_borg_outputs.py --src ../outputs --dst ./outputs --skip-mrfmasked
"""

import argparse
import re
import shutil
from pathlib import Path

RUNTIME_RE = re.compile(
    r"^MMBorg_(?P<islands>\d+)M_(?P<policy>STARFIT|RBF|PWL)_(?P<res>.+)_nfe(?P<nfe>\d+)"
    r"_seed(?P<seed>\d+)"
    r"(?P<mrf>(?:_mrffiltered_(?:regression|perfect)|_mrfmasked(?:_(?:perfect|regression))?))?"
    r"_(?P<island>\d+)\.runtime$"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Directory containing Borg outputs (default: ../outputs relative to this script)",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="MOEA OUT_ROOT layout root (default: ./outputs next to this script)",
    )
    ap.add_argument(
        "--skip-mrfmasked",
        action="store_true",
        help="Skip runtimes with _mrfmasked in the name (use if only running MOEA on full-series seeds)",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    src = (args.src or (here.parent / "outputs")).resolve()
    dst_root = (args.dst or (here / "outputs")).resolve()
    templates = here / "templates"

    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    n_copy = 0
    for path in sorted(src.glob("MMBorg_*.runtime")):
        m = RUNTIME_RE.match(path.name)
        if not m:
            print(f"[skip] unmatched name: {path.name}")
            continue
        if args.skip_mrfmasked and m.group("mrf"):
            continue
        policy = m.group("policy")
        res = m.group("res")
        pol_dir = dst_root / f"Policy_{policy}"
        rt_dir = pol_dir / "runtime" / res
        rt_dir.mkdir(parents=True, exist_ok=True)
        dest = rt_dir / path.name
        shutil.copy2(path, dest)
        n_copy += 1
        print(f"  {path.name} -> {dest.relative_to(dst_root)}")

        hdr_name = f"1-header-file_{policy}.txt"
        hdr_src = templates / hdr_name
        hdr_dst = pol_dir / "1-header-file.txt"
        if hdr_src.is_file() and not hdr_dst.exists():
            shutil.copy2(hdr_src, hdr_dst)
            print(f"  (header) {hdr_src.name} -> {hdr_dst.relative_to(dst_root)}")

    print(f">> Done. Copied {n_copy} runtime files under {dst_root}")


if __name__ == "__main__":
    main()
