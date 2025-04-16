#!/bin/bash

epochs=1000
batch_size=32
max_iters=100000
for seed in $(seq 0 2);
do
    # ./src/script/slate_job.sh shapes3d $seed $batch_size 3 3 1 64 $epochs 4 4 false $max_iters
    sbatch -J slate-shapes3d ./src/script/slate_job.sh shapes3d $seed $batch_size 3 3 1 64 $epochs 4 4 false $max_iters
    # sbatch -J slate-clevr6 ./src/script/slate_job.sh clevr6 $seed $batch_size 7 3 1 128 $epochs 6 6 false $max_iters
    # sbatch -J slate-arrowroom ./src/script/slate_job.sh arrowroom $seed $batch_size 5 3 1 128 $epochs 4 4 false $max_iters

    # ./src/script/slate_job.sh shapes3d $seed $batch_size 3 3 1 64 $epochs 4 4 false
    # sbatch -J slate-shapestack ./src/script/slate_job.sh shapestack $seed $batch_size 12 7 1 96 $epochs 8 8 false
    # sbatch -J slate-celeba ./src/script/slate_job.sh celeba $seed $batch_size 4 3 $epochs 128 1 8 8 true
    # ./src/script/slate_job.sh celeba $seed $batch_size 4 3 1 128 $epochs 8 8 true
    # sbatch -J slate-bitmoji ./src/script/slate_job.sh bitmoji $seed $batch_size 8 3 4 128 $epochs 8 8 false
    # sbatch -J slate-clevrtex ./src/script/slate_job.sh clevrtex $seed $batch_size 12 3 1 128 $epochs 8 8 true

    # sbatch -J slate-objectsroom ./src/script/slate_job.sh objectsroom $seed $batch_size 6 7 1 64 $epochs
done