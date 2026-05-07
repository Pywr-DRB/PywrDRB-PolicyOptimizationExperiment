#!/bin/bash
set -euo pipefail

#SBATCH --job-name=PreprocessObs
#SBATCH --output=logs/preprocess.out
#SBATCH --error=logs/preprocess.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-user=ms3654@cornell.edu

# Run from project root so Python finds the methods package
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

echo "Starting observational data retrieval..."
python 01_retrieve_data.py

echo "Processing raw data..."
INFLOW_SOURCE_MODE="${CEE_INFLOW_SOURCE_MODE:-pub_only}"
echo "Processing raw data with inflow source mode: ${INFLOW_SOURCE_MODE}"
python 02_process_data.py --inflow-source "${INFLOW_SOURCE_MODE}"

echo "Building MRF filtering bundles (regression_disagg + perfect_foresight)..."
bash build_mrf_filtering_folder.sh

echo "Done with preprocessing + MRF filtering build."
