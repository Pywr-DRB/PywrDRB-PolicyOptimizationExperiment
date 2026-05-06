#!/bin/bash
# Quick Slurm view to pick node/task counts before sbatch (Hopper / any Slurm cluster).
# Run from anywhere:  ./scripts/check_slurm_availability.sh
#
# Then submit with matching resources, e.g.:
#   sbatch --nodes=2 --ntasks-per-node=40 run_full_pareto_pywr_mpi.sh

set -euo pipefail

echo "== Host: $(hostname)  User: ${USER:-?}  Date: $(date -Is) =="
echo

if ! command -v sinfo >/dev/null 2>&1; then
  echo "sinfo not found — not a Slurm login node or Slurm not in PATH."
  exit 0
fi

echo "== Partitions (availability summary) =="
# P=partition  a=state  l=timelimit  D=nodes  t=state  c=cpus  free/alloc
sinfo -o "%P %a %l %D %t %C" 2>/dev/null || sinfo
echo

echo "== Idle / mixed / allocated nodes per partition (best-effort) =="
sinfo -h -o "%P %F" 2>/dev/null | while read -r line; do
  echo "  $line"
done
echo "  (Format from %F: a/b/c/d = a=allocated b=idle c=other d=total — exact meaning can vary by site)"
echo

echo "== Your jobs =="
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" 2>/dev/null || squeue -u "${USER}"
else
  echo "squeue not found"
fi
echo

if command -v sshare >/dev/null 2>&1; then
  echo "== Fair-share (if your site exposes sshare) =="
  sshare -u "${USER}" 2>/dev/null || true
  echo
fi

echo "== Suggested sbatch overrides =="
echo "  Total MPI ranks = nodes * ntasks-per-node (one rank per Pywr job)."
echo "  Default job script requests 6 nodes × 40 tasks (240 ranks). Examples:"
echo "    sbatch run_full_pareto_pywr_mpi.sh"
echo "    sbatch --nodes=2 --ntasks-per-node=40 run_full_pareto_pywr_mpi.sh   # lighter"
echo "  Increase nodes if partitions show many idle nodes and your account allows it."
echo
echo "  To see limits for an account/partition (if supported):"
echo "    sacctmgr show qos format=name,maxwall,maxtres%30 2>/dev/null || true"
