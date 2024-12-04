#Type the salloc command below on a Perlmutter login node  to allocate the compute nodes to run your job interactively,  

cd /global/cfs/projectdirs/m4431/DITeTriO/
salloc -N 1 -C gpu -q interactive -t 01:00:00 -G 4 -A m4431


#when the shell prompts, set the environment for your job, e.g., set OpenMP environment variables, load modules, etc., then type the following srun command to run your job interactively. 

#OpenMP settings:
module load python/3.11
module load pytorch/2.0.1
module load pandas/2.2.3
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 --unbuffered python -m model.train
