#!/bin/bash
#SBATCH -p gpus24,gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=slotae
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

export PATH="/vol/biomedic3/rrr2417/.local/bin:$PATH"
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

TRAIN_ARGS="--model_dir runs/slotae --dataset $1 --seed $2 --num_epochs $3"
TRAIN_CMD="poetry run slotae_train $TRAIN_ARGS"
echo $TRAIN_CMD

eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo Train job failed
    exit 1
fi