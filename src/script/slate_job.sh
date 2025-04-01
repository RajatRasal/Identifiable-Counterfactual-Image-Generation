#!/bin/bash
#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

export PATH="/vol/biomedic3/rrr2417/.local/bin:$PATH"
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

TRAIN_ARGS="""
    --dataset $1
    --log_path runs/slate/$1
    --seed $2
    --batch_size $3
    --num_slots $4
    --num_iterations $5
    --num_slot_heads $6
    --image_size $7
    --epochs $8
"""
TRAIN_CMD="poetry run python3 -m models.slate.train $TRAIN_ARGS"
echo $TRAIN_CMD

eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo Train job failed
    exit 1
fi