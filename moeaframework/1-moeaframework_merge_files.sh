#!/usr/bin/env bash
# Converts each *.runtime file to a corresponding *.set file.
# We do this to build references (the .set snapshots are used purely for ref construction).
# The time-series metrics will be computed from *.runtime later.

set -euo pipefail           # strict mode
shopt -s nullglob           # globs that match nothing expand to empty (no literal *)

# ===== CLI autodetect / override =====
CLI_ARG="${1:-}"            # read optional arg
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"; shift 2  # external override
else
  # try common locations
  if   [[ -x "./cli" ]]; then CLI="./cli"
  elif [[ -x "./MOEAFramework-5.0/cli" ]]; then CLI="./MOEAFramework-5.0/cli"
  elif [[ -x "../MOEAFramework-5.0/cli" ]]; then CLI="../MOEAFramework-5.0/cli"
  else echo "ERROR: MOEAFramework cli not found. Pass --cli /path/to/cli" >&2; exit 1; fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: $CLI not executable (chmod +x)"; exit 1; }

# ===== Bounds/filters =====
OUT_ROOT="${OUT_ROOT:-outputs}"       # root folder with Policy_* dirs
SEED_FROM="${SEED_FROM:-1}"           # inclusive seed lower bound
SEED_TO="${SEED_TO:-10}"              # inclusive seed upper bound

echo ">> [1/4] runtime -> set  (seeds ${SEED_FROM}-${SEED_TO})"

# Iterate over every policy directory under OUT_ROOT
for policy_dir in "${OUT_ROOT}"/Policy_*; do
  [[ -d "$policy_dir/runtime" ]] || continue            # skip if no runtime directory
  echo ">> Policy: $(basename "$policy_dir")"

  # For each reservoir (runtime/<reservoir>/ has the .runtime files)
  for rdir in "$policy_dir/runtime"/*/; do
    [[ -d "$rdir" ]] || continue
    reservoir="$(basename "$rdir")"                     # reservoir name
    out_dir="$policy_dir/refsets/$reservoir"            # where we put the converted .set files
    mkdir -p "$out_dir"
    echo "   - Reservoir: $reservoir"

    # For every .runtime file in this reservoir
    for runtime in "$rdir"/*.runtime; do
      base="$(basename "$runtime" .runtime)"            # strip .runtime
      # Expect filenames like ..._seed<seed>_<island>.runtime
      # Optional _mrffiltered[_pub|_perfect] before _<island> (filtered Borg objectives)
      if [[ "$base" =~ _seed([0-9]+)(_mrffiltered(_pub|_perfect)?)?_([0-9]+)$ ]]; then
        seed="${BASH_REMATCH[1]}"                       # captured <seed>
        island="${BASH_REMATCH[4]}"                     # captured <island>
        (( seed < SEED_FROM || seed > SEED_TO )) && continue  # filter seeds not in window
      else
        # If filename doesn't include both seed and island tags, skip strictly
        continue
      fi
      out_set="$out_dir/${base}.set"                    # parallel .set path
      echo "     * $(basename "$runtime") -> $(basename "$out_set")"
      "$CLI" ResultFileConverter --input "$runtime" --output "$out_set"   # convert to .set
    done
  done
done

echo ">> Done."
