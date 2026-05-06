#!/usr/bin/env bash
# Four-step MOEA Framework pipeline: runtime->set, append header, refset merge, metrics.
# Prereq: python organize_borg_outputs.py (from this directory) so ./outputs/Policy_*/runtime/... exists.
#
# Usage (from moeaframework/):
#   export SEED_FROM=1 SEED_TO=10
#   export EPSILON=0.01,0.01,0.01,0.01   # length must match NumberOfObjectives in 1-header-file.txt (4 for current pywrdrb)
#   bash run_moea_workflow.sh
#   bash run_moea_workflow.sh --cli "$(pwd)/MOEAFramework-5.0/cli"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OUT_ROOT="${OUT_ROOT:-outputs}"
export SEED_FROM="${SEED_FROM:-1}"
export SEED_TO="${SEED_TO:-10}"
export EPSILON="${EPSILON:-0.01,0.01,0.01,0.01}"
export NUM_OBJS="${NUM_OBJS:-4}"

if [[ "${1:-}" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"
else
  if   [[ -x "$SCRIPT_DIR/MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/MOEAFramework-5.0/cli"
  elif [[ -x "$SCRIPT_DIR/cli" ]]; then CLI="$SCRIPT_DIR/cli"
  else
    echo "ERROR: MOEAFramework cli not found. Install MOEAFramework-5.0 or pass --cli /path/to/cli" >&2
    exit 1
  fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: not executable: $CLI"; exit 1; }

echo "== MOEA workflow =="
echo "CLI:       $CLI"
echo "OUT_ROOT:  $OUT_ROOT"
echo "SEEDS:     $SEED_FROM-$SEED_TO"
echo "EPSILON:   $EPSILON"
echo

echo ">> [1/4] runtime -> .set"
bash "$SCRIPT_DIR/1-moeaframework_merge_files.sh" --cli "$CLI"
echo

echo ">> [2/4] append header -> *_header.set"
python3 "$SCRIPT_DIR/append_header.py" --outputs-root "$OUT_ROOT" --seed-from "$SEED_FROM" --seed-to "$SEED_TO"
echo

echo ">> [3/4] merge -> seed*.ref and <reservoir>.ref"
EPSILON="$EPSILON" SEED_FROM="$SEED_FROM" SEED_TO="$SEED_TO" \
  bash "$SCRIPT_DIR/2-moeaframework_gen_refset.sh" --cli "$CLI"
echo

echo ">> [4/4] MetricsEvaluator vs global .ref"
EPSILON="$EPSILON" SEED_FROM="$SEED_FROM" SEED_TO="$SEED_TO" \
  bash "$SCRIPT_DIR/3-moeaframework_gen_runtime.sh" --cli "$CLI"
echo
echo "== Done. Metrics under ${OUT_ROOT}/Policy_*/metrics/ =="
