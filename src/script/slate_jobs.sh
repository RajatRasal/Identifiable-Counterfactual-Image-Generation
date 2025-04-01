#!/bin/bash

epochs=200
batch_size=64
for seed in $(seq 0 0);
do
    # sbatch -J slate-shapes3d ./src/script/slate_job.sh shapes3d $seed $batch_size 3 3 1 64 $epochs
    # sbatch -J slate-objectsroom ./src/script/slate_job.sh objectsroom $seed $batch_size 6 7 1 64 $epochs
    # sbatch -J slate-clevr6 ./src/script/slate_job.sh clevr6 $seed $batch_size 7 7 1 128 $epochs
    sbatch -J slate-arrowroom ./src/script/slate_job.sh arrowroom $seed $batch_size 5 7 1 128 $epochs
    # sbatch -J slate-shapestack ./src/script/slate_job.sh shapestack $seed $batch_size 12 7 1 128 $epochs
    # sbatch -J slate-bitmoji ./src/script/slate_job.sh bitmoji $seed $batch_size 8 3 4 128 $epochs
    # sbatch -J slate-clevrtex ./src/script/slate_job.sh clevrtex $seed $batch_size 12 3 1 128 $epochs
    # sbatch -J slate-celeba ./src/script/slate_job.sh celeba $seed $batch_size 4 3 1 128 $epochs
done