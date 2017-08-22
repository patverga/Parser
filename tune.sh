#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

num_gpus=108

lrs="0.04 0.08 0.06 0.02 0.1 0.01"
mus="0.9"
nus="0.9, 0.98"
epsilons="1e-12 1e-8 1e-4"
warmup_steps="4000 2000 8000"
batch_sizes="5000 2048 1024"
reps="3"

# 6*2*3*3*3*3



# array to hold all the commands we'll distribute
declare -a commands

for lr in ${lrs[@]}; do
    for mu in ${mus[@]}; do
        for nu in ${nus[@]}; do
            for epsilon in ${epsilons[@]}; do
                for warmup_steps in ${warmup_steps[@]}; do
                    for batch_size in ${batch_sizes[@]}; do
                        for rep in `seq $reps`; do
                            fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size"
                            commands+=("srun --gres=gpu:1 --partition=titanx-short python network.py \
                            --config_file config/myconf.cfg \
                            --save_dir $OUT_LOG/scores-$fname_append \
                            --save_every 500 \
                            --train_iters 100000 \
                            --train_batch_size $batch_size \
                            --warmup_steps $warmup_steps \
                            --learning_rate $lr \
                            --mu $mu \
                            --nu $nu \
                            --epsilon $epsilon \
                            &> $OUT_LOG/train-$fname_append.log")
                        done
                    done
                done
            done
        done
    done
done

# now distribute them to the gpus
num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for (( gpuid=0; gpuid<num_gpus; gpuid++)); do
    for (( i=0; i<jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]}"
        comm=${comm/XX/$gpuid}
#        echo "Starting job $jobid on gpu $gpuid"
        echo ${comm}
        if [[ "$debug" == "false" ]]; then
            eval ${comm}
        fi
    done &
    j=$((j + 1))
done
