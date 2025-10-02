#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64  
# comment ahha --mem-per-cpu=8192M   # memory per CPU core
#SBATCH --mem=128G
#SBATCH -J "MVO-bt"   # job name
#SBATCH --mail-user=porter77@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/porter77/sf_fall_2025/sf-quant-labs/logs/mombt_%j.out  # Output log file
#SBATCH --error=/home/porter77/sf_fall_2025/sf-quant-labs/logs/mombt_%j.err  # Error log file
#SBATCH --qos=standby


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
# export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

export OMP_NUM_THREADS=1

export ASSETS_TABLE="/home/porter77/groups/grp_quant/database/research/assets"
export EXPOSURES_TABLE='/home/porter77/groups/grp_quant/database/research/exposures' #in grp_quant the folder is exposures without the "crsp_" part
export COVARIANCES_TABLE='/home/porter77/groups/grp_quant/database/research/covariances' #same here

# Activate virtual environment
source /home/porter77/sf_fall_2025/sf-quant-labs/.venv/bin/activate
# Navigate to the project directory
cd /home/porter77/sf_fall_2025/sf-quant-labs/labs/

# Run the backtest script using all available cores
python mvo.py
