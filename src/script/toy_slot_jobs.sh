#!/bin/bash

epochs=501
for seed in $(seq 0 2);
do
    sbatch ./src/script/toy_slot_job.sh clevr6 $seed $epochs
    sbatch ./src/script/toy_slot_job.sh arrowroom $seed $epochs
done