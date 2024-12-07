#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -q regular
#SBATCH -J Tetris
#SBATCH --mail-user=bhc31@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -A m4431_g
#SBATCH --output=/global/cfs/projectdirs/m4431/DITeTriO/model/slurm_out/slurm-%j.out


#OpenMP settings:
cd /global/cfs/projectdirs/m4431/DITeTriO/
module load python/3.11
module load pytorch/2.0.1
# source /global/cfs/projectdirs/m4431/DITeTriO/tetr_env/bin/activate
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train $TRAIN_ARGS
