#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00
#SBATCH --partition normal-a100-40
#SBATCH --gpus=1
#SBATCH --account=f202500010hpcvlabuminhog

ml CUDA

make clean
make

srun -n1 ./zpic
