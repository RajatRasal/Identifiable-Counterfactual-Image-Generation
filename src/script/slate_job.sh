#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

export PATH="/vol/biomedic3/rrr2417/.local/bin:$PATH"
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

TRAIN_ARGS="""
    --dataset $1
    --log_path runs/slate/$1_$2
    --seed $2
    --batch_size $3
    --num_slots $4
    --num_iterations $5
    --num_slot_heads $6
    --image_size $7
    --epochs $8
    --num_dec_blocks $9
    --num_heads ${10}
    --max_iters ${12}
"""
TRAIN_CMD="poetry run slate_train $TRAIN_ARGS"
if ${11}; then
    TRAIN_CMD="$TRAIN_CMD --latent_cnn"
fi
echo $TRAIN_CMD

eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo Train job failed
    exit 1
fi