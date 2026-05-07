#!/usr/bin/env python3
# Minimal “append header” for refsets:
# For each Policy_*/refsets/<reservoir>/*_seedS_M.set:
#   new file = *_seedS_M_header.set with the policy header prepended
#   (first two lines of the original are dropped).
#
# Extra tweak: if a data row has NV+NO values and NC>0, append NC zeros.

import argparse, sys
from pathlib import Path
import glob

def parse_int_from_header(lines, tag):
    for ln in lines:
        if ln.startswith(f"# {tag}="):
            try:
                return int(ln.split("=",1)[1].strip())
            except Exception:
                pass
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--seed-from", type=int, default=1)
    ap.add_argument("--seed-to", type=int, default=10)
    ap.add_argument("--num-masters", type=int, default=4)        # islands per seed
    ap.add_argument("--policies-filter", default="Policy_*")      # optional: narrow to one policy
    ap.add_argument("--reservoirs-filter", default="*")           # optional: narrow to one reservoir
    args = ap.parse_args()

    OUT = Path(args.outputs_root)

    print(">> append header -> *_header.set")
    print(f"Seeds {args.seed_from}-{args.seed_to} | Masters 0..{args.num_masters-1}")

    for policy_dir in sorted(OUT.glob(args.policies_filter)):
        if not policy_dir.is_dir():
            continue
        header_path = policy_dir / "1-header-file.txt"
        refsets_root = policy_dir / "refsets"
        if not header_path.exists() or not refsets_root.exists():
            continue

        header_lines = header_path.read_text(encoding="utf-8").strip().splitlines()

        # read NV/NO/NC from header to know when to pad
        NV = parse_int_from_header(header_lines, "NumberOfVariables")
        NO = parse_int_from_header(header_lines, "NumberOfObjectives")
        NC = parse_int_from_header(header_lines, "NumberOfConstraints")

        print(f">> {policy_dir.name} | Header: {header_path.name}")

        for rdir in sorted(refsets_root.glob(args.reservoirs_filter)):
            if not rdir.is_dir():
                continue
            wrote_any = False

            for seed in range(args.seed_from, args.seed_to + 1):
                for master in range(args.num_masters):
                    # match *anything*_seed<seed>_<master>.set in this reservoir
                    patterns = [
                        str(rdir / f"*_*_seed{seed}_{master}.set"),
                        str(rdir / f"*_*_seed{seed}_mrffiltered_regression_{master}.set"),
                        str(rdir / f"*_*_seed{seed}_mrffiltered_perfect_{master}.set"),
                    ]
                    matched = []
                    for pattern in patterns:
                        matched.extend(glob.glob(pattern))
                    for set_path_str in sorted(set(matched)):
                        set_path = Path(set_path_str)
                        out_path = set_path.with_name(set_path.stem + "_header.set")

                        set_lines = set_path.read_text(encoding="utf-8").splitlines()

                        # keep payload from line 3+; (original behavior)
                        payload = set_lines[2:] if len(set_lines) >= 2 else []

                        # minimal tweak: pad NC zeros if row has NV+NO values
                        body = []
                        for ln in payload:
                            if not ln.strip():
                                continue
                            cols = ln.split()
                            if NC > 0 and len(cols) == (NV + NO):
                                cols += ["0.0"] * NC
                                ln = " ".join(cols)
                            body.append(ln)

                        # build entry: header + metadata line + body + trailing '#'
                        meta = f"//seed={seed};master={master};file={set_path.name}"
                        lines_to_write = header_lines + [meta] + body + ["#"]
                        content = "\n".join(lines_to_write) + "\n"

                        out_path.write_text(content, encoding="utf-8")
                        print(f"   - {rdir.name}: {set_path.name} -> {out_path.name}")
                        wrote_any = True

            if not wrote_any:
                print(f"   - {rdir.name}: (no matching .set files for the requested seeds/masters)")

    print(">> Done.")

if __name__ == "__main__":
    sys.exit(main())
