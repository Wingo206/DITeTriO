#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -q debug
#SBATCH -J Tetris
#SBATCH --mail-user=bc780@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:15:00
#SBATCH -A m4431_g

#OpenMP settings:
module load python/3.11
module load pytorch/2.0.1
module load pandas/2.2.3
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 8 -c 32 --cpu_bind=cores -G 8 --gpu-bind=single:1 python train.py