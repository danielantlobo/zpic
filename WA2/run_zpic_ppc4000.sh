#!/bin/bash
# WA2: CUDA run at 4,000 particles per cell (the report's headline row), with GPU model logged.
#SBATCH --job-name=zpic-a100-ppc4000
#SBATCH --partition=normal-a100-40
#SBATCH --account=f202500010hpcvlabuminhog
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:05:00

ml CUDA

mkdir -p results
LOG=results/ppc4000_${SLURM_JOB_ID}.log

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$LOG"
nvcc --version | tail -n 2 | tee -a "$LOG"

make clean && make PPC=4000
srun -n1 ./zpic | tee -a "$LOG"
