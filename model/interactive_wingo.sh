#Type the salloc command below on a Perlmutter login node  to allocate the compute nodes to run your job interactively,  

cd /global/cfs/projectdirs/m4431/DITeTriO/
salloc -N 2 -C gpu -q interactive -t 01:00:00 -G 8 -A m4431
salloc -N 1 -C gpu -q interactive -t 01:00:00 -G 2 -A m4431


#when the shell prompts, set the environment for your job, e.g., set OpenMP environment variables, load modules, etc., then type the following srun command to run your job interactively. 

#OpenMP settings:
module load python/3.11
module load pytorch/2.0.1
source /global/cfs/projectdirs/m4431/DITeTriO/tetr_env/bin/activate
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 8 -c 32 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered python -m model.train --train_data "data/processed_replays/players/caboozled_pie/*.csv" --epochs 100 --output_dir "$SCRATCH/tetris/output"

srun -n 2 --cpu_bind=cores -G 2 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train \
    --train_data "data/processed_replays/players/sodiumoverdose/*.csv" \
    --epochs 25 --batch 1000 \
    --output_dir "$SCRATCH/tetris/outputtest" \
    --conv_channels 16 32 48 \
    --conv_kernels 5 3 3 \
    --linears 100 100
