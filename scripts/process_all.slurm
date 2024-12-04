#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -J process_replays
#SBATCH --mail-user=bhc31@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
python /global/cfs/cdirs/m4431/DITeTriO/scripts/organize.py --start $START --end $END

