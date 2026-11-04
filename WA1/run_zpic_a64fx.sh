#!/bin/bash
# WA1: A64FX OpenMP run, as measured in report.pdf (12 threads bound to one CMG).
#SBATCH --job-name=zpic-a64fx
#SBATCH --partition=normal-arm
#SBATCH --account=F202500010HPCVLABUMINHOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --exclusive            # whole node: no other job shares the CMG or HBM2
#SBATCH --time=00:05:00

ml GCC/13.3.0

make clean && make

export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active

srun --cpus-per-task=12 ./zpic
