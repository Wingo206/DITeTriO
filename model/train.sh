#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q preempt
#SBATCH -J ditetrio_debug
#SBATCH --mail-user=bc780@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00
#SBATCH -A m4331_g

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 4 -c 32 --cpu_bind=cores -G 5 --gpu-bind=single:1 python train.py
