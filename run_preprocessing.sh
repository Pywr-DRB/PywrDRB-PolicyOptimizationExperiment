#!/bin/bash
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
python 02_process_data.py

echo "Done with preprocessing."
