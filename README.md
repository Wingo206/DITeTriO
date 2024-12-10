# DITeTriO

# run interactive training
cd /global/cfs/projectdirs/m4431/DITeTriO/
salloc -N 2 -C gpu -q interactive -t 01:00:00 -G 8 -A m4431
module load python/3.11
module load pytorch/2.0.1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# run interactive training (small)
cd /global/cfs/projectdirs/m4431/DITeTriO/
salloc -N 1 -C gpu -q shared_interactive -t 01:00:00 -G 2 -A m4431
module load python/3.11
module load pytorch/2.0.1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/15_bigcnn_longo
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.00 \
--remove_last_frame True
EOF
)
srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train $TRAIN_ARGS

# submit batch
cd /global/cfs/projectdirs/m4431/DITeTriO/
sbatch --export=ALL,TRAIN_ARGS="$TRAIN_ARGS" /global/cfs/projectdirs/m4431/DITeTriO/model/train_wingo.sh


# run preprocessor:
go to DITeTriOPreProcessor directory and do ```dotnet run```
compile executable: ``````

# processing everything
cd /global/cfs/projectdirs/m4431/DITeTriO/scripts
sbatch --export=START=0,END=500 process_all.slurm

**make sure you run the next script after all the processing, which delets empty files**
python remove_empty.py --directory ../data/processed_replays/players

# evaluate w/ remote desktop:
module load python/3.11
module load pytorch/2.0.1
cd /global/cfs/projectdirs/m4431/DITeTriO/
source tetr_env/bin/activate
python -m scripts.evaluate_model --model_dir saved_models/nolastframe/1_bigcnn_longo --train_data data/processed_replays/players/sodiumoverdose/6657e2e7cdcf03ad6260a6d8_p1_r0.csv --remove_last_frame True

# simulation w/ remote desktop:
upen up a tmux, so both processes are on the same node
ctrl + shift + v to paste in the remote desktop

module load python/3.11
module load pytorch/2.0.1
cd /global/cfs/projectdirs/m4431/DITeTriO/
source tetr_env/bin/activate
python -m scripts.simulate_model --pipedir /tmp --replay  /global/cfs/projectdirs/m4431/DITeTriO/data/top100/data/66920278bafbc037c5c3d177.ttrm --model_dir saved_models/nolastframe/1_bigcnn_longo --remove_last_frame True

---on second terminal:
cd /global/cfs/projectdirs/m4431/DITeTriO/DITeTriO_cs/TetrEnvLink
dotnet run "/tmp"

