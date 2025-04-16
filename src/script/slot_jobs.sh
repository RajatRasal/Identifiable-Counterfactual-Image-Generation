#!/bin/bash

epochs=1000
batch_size=32
max_iters=100000
for seed in $(seq 0 2);
do
    sbatch -J slotae-shapes3d ./src/script/slot_job.sh shapes3d $seed $epochs 64 $batch_size 3 3 $max_iters
    # sbatch -J slotae-clevr6 ./src/script/slot_job.sh clevr6 $seed $epochs 128 $batch_size 7 3 $max_iters
    # sbatch -J slotae-arrowroom ./src/script/slot_job.sh arrowroom $seed $epochs 128 $batch_size 5 3 $max_iters

    # sbatch -J slotae-objectsroom ./src/script/slot_job.sh objectsroom $seed $epochs $resolution $batch_size 6 3
done