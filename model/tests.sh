# interactive running
cd /global/cfs/projectdirs/m4431/DITeTriO/
salloc -N 2 -C gpu -q interactive -t 01:00:00 -G 8 -A m4431

module load python/3.11
module load pytorch/2.0.1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train $TRAIN_ARGS

# debug sbatch
sbatch --export=ALL,TRAIN_ARGS="$TRAIN_ARGS" /global/cfs/projectdirs/m4431/DITeTriO/model/train_debug_wingo.sh

# train sbatch
sbatch --export=ALL,TRAIN_ARGS="$TRAIN_ARGS" /global/cfs/projectdirs/m4431/DITeTriO/model/train_wingo.sh



# Interactive tests
# way overfit
srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train \
    --train_data "data/processed_replays/players/sodiumoverdose/*.csv" \
    --epochs 100 --batch 1000 \
    --output_dir "$SCRATCH/tetris/outputhuge" \
    --conv_channels 16 24 32 48 64 \
    --conv_kernels 5 5 3 3 3 \
    --linears 1000 500 100 100

srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train \
    --train_data "data/processed_replays/players/sodiumoverdose/*.csv" \
    --epochs 100 --batch 1000 \
    --output_dir "$SCRATCH/tetris/less_padding_dropout_bigger_model" \
    --conv_channels 16 24 32 48 64 \
    --conv_kernels 5 5 3 3 3 \
    --conv_padding 2 1 0 0 0 \
    --linears 1000 500 100 100 \
    --dropouts 0.5 0.5 0.5 0.5

srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train \
    --train_data "data/processed_replays/players/sodiumoverdose/*.csv" \
    --epochs 100 --batch 1000 \
    --output_dir "$SCRATCH/tetris/small_model_no_dropout_last" \
    --conv_channels 16 24 32 \
    --conv_kernels 5 3 3 \
    --conv_padding 2 1 0 \
    --linears 200 100 \
    --dropouts 0.5 0.5


# should be good
TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir $SCRATCH/tetris/lots_of_conv \
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 \
--dropouts 0.5 0.5
EOF
)
sbatch --export=ALL,TRAIN_ARGS="$TRAIN_ARGS" train_debug_wingo.sh


TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir $SCRATCH/tetris/small_dropout_fixed \
--conv_channels 16 24 32 \
--conv_kernels 5 3 3 \
--conv_padding 2 1 0 \
--linears 100 100 \
--dropouts 0.5 0.5
EOF
)
srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train $TRAIN_ARGS

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir $SCRATCH/tetris/brondong \
--conv_channels 16 24 32 48 \
--conv_kernels 5 3 3 3 \
--conv_padding 2 1 0 0 \
--linears 500 200 100 \
--dropouts 0.5 0. batch.size(0)
srun -n 8 -c 16 --cpu_bind=cores -G 8 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train $TRAIN_ARGS

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/2_smallo
--conv_channels 16 24 32 \
--conv_kernels 5 3 3 \
--conv_padding 2 1 1 \
--linears 200 100 50 \
--dropouts 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/jaazen/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/testo
--conv_channels 16 24 32 \
--conv_kernels 5 3 3 \
--conv_padding 2 1 1 \
--linears 200 100 50 \
--dropouts 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/7_BCE_Thingy
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 500 200 100 \
--dropouts 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/10_superupersmallo
--conv_channels 4 4 \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 100 50 \
--dropouts 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/12_sigmoid_longo_no_weightsample
--conv_channels 4 4 \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/9_ddp_mse
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 500 200 100 \
--dropouts 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/13_BCE_longo
--conv_channels 4 4 \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)


TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/14_longcnn_longo
--conv_channels 8 8 8 8 8 8 8 8 \
--conv_kernels 3 3 3 3 3 3 3 3 \
--conv_padding 1 1 1 1 1 1 1 1 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/15_bigcnn_longo
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/16_bigcnn_longo_10kbatch
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/16_lesschannelscnn_longo_10kbatch
--conv_channels 16 16 16 16 16 16 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/18_bigcnn_longo_laterdropout
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0.2 0.2 0.2 0.2 0.2 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/19_smallmodel_100k
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 100 100 \
--dropouts 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/20_smallmodel_10k
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 100 100 \
--dropouts 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/21_smallcnn_mediumlinear_100k
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 100 100 100 100 100 \
--dropouts 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/22_smallcnn_mediumlinear_10k
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 100 100 100 100 100 \
--dropouts 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/23_smallcnn_mediumlinear2_100k
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 500 200 100 100 100 \
--dropouts 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/24_smallcnn_mediumlinear2_10k
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 500 200 100 100 100 \
--dropouts 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/25_smallcnn_longlinearbeefy_100k
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 500 200 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/26_smallcnn_longlinearbeefy_100k_halfdropout
--conv_channels 8 8  \
--conv_kernels 3 3 \
--conv_padding 1 1 \
--linears 500 200 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/5han/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/5han/1_bigcnn_longo_dropout25
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0.25 0.25 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/5han/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/5han/2_10k_bigcnn_longo_dropout25
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0.25 0.25 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/5han/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/5han/3_100k_bigcnn_longo
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/5han/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/5han/4_100k_bigcnn_longo_beefer
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 500 200 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/5han/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/5han/5_100k_bigcnn_longo_beefer_dropout
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 500 200 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0.25 0.25 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/5han/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/5han/6_longcnn_longo_dropout
--conv_channels 8 8 8 8 8 8 8 8 \
--conv_kernels 3 3 3 3 3 3 3 3 \
--conv_padding 1 1 1 1 1 1 1 1 \
--linears 500 200 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0.25 0.25 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001
EOF
)
TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/5han/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/5han/7_longcnn_longo
--conv_channels 8 8 8 8 8 8 8 8 \
--conv_kernels 3 3 3 3 3 3 3 3 \
--conv_padding 1 1 1 1 1 1 1 1 \
--linears 500 200 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/nolastframe/1_bigcnn_longo
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001 \
--remove_last_frame True
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/nolastframe/2_smaller
--conv_channels 16 24 32 \
--conv_kernels 5 5 3 \
--conv_padding 2 2 0 \
--linears 100 100 100 100 \
--dropouts 0 0 0 0 \
--lr 0.001 \
--remove_last_frame True
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/nolastframe/3_bigcnn_longo_10k
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0 0 0 0 0 \
--lr 0.001 \
--remove_last_frame True
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/nolastframe/4_bigcnn_longo_10k_dropout
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 100 100 100 100 100 100 100 100 100 100 \
--dropouts 0 0 0 0 0 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001 \
--remove_last_frame True
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/nolastframe/5_smaller_10k
--conv_channels 16 24 32 \
--conv_kernels 5 5 3 \
--conv_padding 2 2 0 \
--linears 500 200 100 \
--dropouts 0 0 0 \
--lr 0.001 \
--remove_last_frame True
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 10000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/nolastframe/6_biggerlinearbiggo_dropout
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 200 100 100 100 100 100 100 \
--dropouts 0 0 0 0.25 0.25 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001 \
--remove_last_frame True
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 100000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/nolastframe/6_biggerlinearbiggo_dropout_100k
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 200 100 100 100 100 100 100 \
--dropouts 0 0 0 0.25 0.25 0.25 0.25 0.25 0.25 0.25 \
--lr 0.001 \
--remove_last_frame True
EOF
)

5han test
# simulation w/ remote desktop:
upen up a tmux, so both processes are on the same node
ctrl + shift + v to paste in the remote desktop

module load python/3.11
module load pytorch/2.0.1
cd /global/cfs/projectdirs/m4431/DITeTriO/
source tetr_env/bin/activate
python -m scripts.simulate_model --pipedir /tmp --replay  /global/cfs/projectdirs/m4431/DITeTriO/data/top100/data/66a34a3ee3e9b74a38ce5bf4.ttrm --model_dir saved_models/5han/2_10k_bigcnn_longo_dropout25

---on second terminal:
cd /global/cfs/projectdirs/m4431/DITeTriO/DITeTriO_cs/TetrEnvLink
dotnet run "/tmp"

----------------------LSTM testing---------------

# evaluate w/ remote desktop:
module load python/3.11
module load pytorch/2.0.1
cd /global/cfs/projectdirs/m4431/DITeTriO/
source tetr_env/bin/activate
python -m scripts.evaluate_model --model_dir saved_models/lstm/3_bigger --train_data data/processed_replays/players/sodiumoverdose/6657e2e7cdcf03ad6260a6d8_p1_r0.csv --lstm True --seq_len 100

# simulation w/ remote desktop:
upen up a tmux, so both processes are on the same node
ctrl + shift + v to paste in the remote desktop

module load python/3.11
module load pytorch/2.0.1
cd /global/cfs/projectdirs/m4431/DITeTriO/
source tetr_env/bin/activate
python -m scripts.simulate_model --pipedir /tmp --replay  /global/cfs/projectdirs/m4431/DITeTriO/data/top100/data/66920278bafbc037c5c3d177.ttrm --model_dir saved_models/lstm/3_bigger --lstm True --seq_len 100

---on second terminal:
cd /global/cfs/projectdirs/m4431/DITeTriO/DITeTriO_cs/TetrEnvLink
dotnet run "/tmp" 1 1

# small data
TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/linustechtips/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/1_firstlstm
--conv_channels 8 8 8 \
--conv_kernels 3 3 3 \
--conv_padding 1 1 1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0 \
--linears 100 100 \
--dropouts 0 0 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/1_firstlstm
--conv_channels 8 8 8 \
--conv_kernels 3 3 3 \
--conv_padding 1 1 1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0 \
--linears 100 100 \
--dropouts 0 0 \
--lr 0.001 
EOF
)

# small data
TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/linustechtips/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/sched_samp_test
--conv_channels 8 8 8 \
--conv_kernels 3 3 3 \
--conv_padding 1 1 1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0 \
--sched_samp True \
--linears 100 100 \
--dropouts 0 0 \
--lr 0.001 
EOF
)
cd /global/cfs/projectdirs/m4431/DITeTriO/
srun -n 1 -c 2 --cpu_bind=cores -G 1 --gpu-bind=single:1 --unbuffered --signal=SIGTERM@2 python -m model.train $TRAIN_ARGS

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/2_sched_samp
--conv_channels 8 8 8 \
--conv_kernels 3 3 3 \
--conv_padding 1 1 1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0 \
--sched_samp True \
--linears 100 100 \
--dropouts 0 0 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/2_sched_samp
--conv_channels 8 8 8 \
--conv_kernels 3 3 3 \
--conv_padding 1 1 1 \
--lstm True --seq_len 10 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0 \
--sched_samp True \
--linears 100 100 \
--dropouts 0 0 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/3_bigger
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 15 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 100 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/4_bigger_BCE
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 15 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/5_BCE_10seq
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 10 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/6_BCE_morelstmlayer
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 4 --lstm_hidden_size 100 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/7_BCE_lesslstmlayer
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 1 --lstm_hidden_size 100 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/8_BCE_morelstmdense
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 500 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/9_BCE_lesslstmdense
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 50 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/10_BCE_10seq_lesslstmdense
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 10 \
--lstm_layers 2 --lstm_hidden_size 50 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/11_BCE_10seq_lesslstmdense_less_layer
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0.1 0.1 \
--lstm True --seq_len 10 \
--lstm_layers 1 --lstm_hidden_size 50 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/12_bigger_BCE_nodropout
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 \
--dropouts 0 0 0 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)


TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/13_bigger_BCE_more_linear
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 100 100 100 100 \
--dropouts 0 0.1 0.1 0.1 0.1 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 2 --lstm_hidden_size 100 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)

TRAIN_ARGS=$(cat <<EOF
--train_data "/global/cfs/projectdirs/m4431/DITeTriO/data/processed_replays/players/sodiumoverdose/*.csv" \
--epochs 50 --batch 1000 \
--output_dir /global/cfs/projectdirs/m4431/DITeTriO/saved_models/lstm/14_less_lstmlayerdense \
--conv_channels 16 24 32 48 64 96 \
--conv_kernels 5 5 3 3 3 3 \
--conv_padding 2 2 0 0 0 0 \
--linears 1000 500 200 100 100 100 100 \
--dropouts 0 0.1 0.1 0.1 0.1 0.1 0.1 \
--lstm True --seq_len 100 \
--lstm_layers 1 --lstm_hidden_size 50 --lstm_dropout 0.1 \
--sched_samp True \
--ramp_epochs 25 \
--lr 0.001 
EOF
)
