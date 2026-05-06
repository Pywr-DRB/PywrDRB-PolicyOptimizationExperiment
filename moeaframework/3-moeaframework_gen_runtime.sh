#!/usr/bin/env bash
# Computes runtime metrics per SEED and per MASTER (island) using MOEAFramework.
# Non-destructive: if a runtime lacks a v5 header, we create a temp file with the header
# and feed that to the CLI. Original files are never modified.

set -euo pipefail
shopt -s nullglob

# ===== CLI autodetect / override =====
CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"; shift 2
else
  if   [[ -x "./cli" ]]; then CLI="./cli"
  elif [[ -x "./MOEAFramework-5.0/cli" ]]; then CLI="./MOEAFramework-5.0/cli"
  elif [[ -x "../MOEAFramework-5.0/cli" ]] ; then CLI="../MOEAFramework-5.0/cli"
  else echo "ERROR: Could not find MOEAFramework cli. Pass --cli /path/to/cli" >&2; exit 1; fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: $CLI not executable (chmod +x)"; exit 1; }

# ===== Config (env-overridable) =====
OUT_ROOT="${OUT_ROOT:-outputs}"
EPSILON="${EPSILON:-0.01,0.01,0.01,0.01}"
SEED_FROM="${SEED_FROM:-1}"
SEED_TO="${SEED_TO:-10}"
NUM_MASTERS="${NUM_MASTERS:-4}"         # islands indexed 0..NUM_MASTERS-1

# ===== Helpers =====
get_problem () { awk -F= '/^# *Problem=/{gsub(/ /,"",$2);print $2;exit}'            "$1/1-header-file.txt"; }
get_no      () { awk -F= '/^# *NumberOfObjectives=/{gsub(/ /,"",$2);print $2;exit}'  "$1/1-header-file.txt"; }
eps_count   () { awk -F, '{print NF}'; }

echo ">> [4/4] Metrics: RUNTIME vs GLOBAL REF"
echo "Seeds: ${SEED_FROM}-${SEED_TO} | Masters (islands): 0..$((NUM_MASTERS-1))"
echo "EPSILON: $EPSILON"
echo

# ===== Per-seed, per-master loop =====
for s in $(seq "$SEED_FROM" "$SEED_TO"); do
  for master in $(seq 0 $((NUM_MASTERS-1))); do
    echo "== Seed $s | Master $master =="

    for policy_dir in "${OUT_ROOT}"/Policy_*; do
      [[ -d "$policy_dir/runtime" ]] || continue

      PROBLEM="$(get_problem "$policy_dir" || true)"
      NO="$(get_no "$policy_dir" || true)"
      if [[ -z "${PROBLEM:-}" || -z "${NO:-}" ]]; then
        echo "   !! Skip $(basename "$policy_dir"): invalid or missing 1-header-file.txt"
        continue
      fi

      # Verify epsilon length matches NO for this policy
      if [[ "$(eps_count <<<"$EPSILON")" -ne "$NO" ]]; then
        echo "   !! Skip $(basename "$policy_dir"): EPSILON has $(eps_count <<<"$EPSILON") entries; NO=$NO"
        continue
      fi

      HEADER_FILE="$policy_dir/1-header-file.txt"

      for rdir in "$policy_dir/runtime"/*/; do
        [[ -d "$rdir" ]] || continue
        reservoir="$(basename "$rdir")"

        ref_global="$policy_dir/refsets/$reservoir/${reservoir}.ref"
        if [[ ! -f "$ref_global" ]]; then
          echo "   - Reservoir: $reservoir  (SKIP: missing ref $ref_global)"
          continue
        fi

        metrics_dir="$policy_dir/metrics/$reservoir"
        mkdir -p "$metrics_dir"

        echo "   - Reservoir: $reservoir"
        echo "     Reference: $(command -v realpath >/dev/null 2>&1 && realpath "$ref_global" || echo "$ref_global")"

        runfiles=( )
        runfiles+=( "$rdir"/*_seed${s}_${master}.runtime )
        runfiles+=( "$rdir"/*_seed${s}_mrffiltered_regression_${master}.runtime )
        runfiles+=( "$rdir"/*_seed${s}_mrffiltered_perfect_${master}.runtime )
        runfiles+=( "$rdir"/*_seed${s}_mrfmasked_${master}.runtime )
        runfiles+=( "$rdir"/*_seed${s}_mrfmasked_perfect_${master}.runtime )
        runfiles+=( "$rdir"/*_seed${s}_mrfmasked_regression_${master}.runtime )
        [[ ${#runfiles[@]} -gt 0 ]] || { echo "     (no runtimes for seed ${s}, master ${master})"; continue; }

        for infile in "${runfiles[@]}"; do
          base="$(basename "$infile" .runtime)"
          outfile="$metrics_dir/${base}.metric"

          # Decide what to feed to the CLI
          input_for_cli="$infile"
          tmp_runtime=""
          if ! grep -q "^# Version=5" "$infile"; then
            echo "     + Prepending header (temp) for $(basename "$infile")"
            tmp_runtime="$(mktemp)"
            { cat "$HEADER_FILE"; echo; cat "$infile"; } > "$tmp_runtime"
            input_for_cli="$tmp_runtime"
          fi

          echo "     * $(basename "$infile") -> $(basename "$outfile")"
          rm -f "$outfile"

          "$CLI" MetricsEvaluator \
            --problem "$PROBLEM" \
            --epsilon "$EPSILON" \
            --input "$input_for_cli" \
            --output "$outfile" \
            --reference "$ref_global"

          # Normalize metric header (remove leading '# ' if present)
          [[ -f "$outfile" ]] && sed -i '1s/^#\s*//' "$outfile" || true

          # Safe cleanup
          if [[ -n "$tmp_runtime" ]]; then rm -f "$tmp_runtime"; fi
        done
      done
    done
    echo
  done
done

echo ">> Done."
