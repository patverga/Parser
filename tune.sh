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

lrs="0.00075 0.0005 0.001"
mus="0.9 0.1"
nus="0.9"
epsilons="1e-12 1e-8 1e-4"
decays="0.75 0.9"
decay_step_vals="1000 2500 5000 50000"
batch_sizes="5000 1500"
reps="3"

# 3*3*3*3*4*3*2 = 1944

# array to hold all the commands we'll distribute
declare -a commands

for lr in ${lrs[@]}; do
    for mu in ${mus[@]}; do
        for nu in ${nus[@]}; do
            for epsilon in ${epsilons[@]}; do
                for decay in ${decays[@]}; do
                    for decay_steps in ${decay_step_vals[@]}; do
                        for batch_size in ${batch_sizes[@]}; do
                            for rep in `seq $reps`; do
                                fname_append="$rep-$lr-$mu-$nu-$epsilon-$decay-$decay_steps-$batch_size"
                                commands+=("srun --gres=gpu:1 --partition=titanx-short python network.py \
                                --config_file config/myconf.cfg \
                                --save_dir $OUT_LOG/scores-$fname_append \
                                --save_every 500 \
                                --decay $decay \
                                --train_batch_size $batch_size \
                                --decay_steps $decay_steps \
                                --learning_rate $learning_rate \
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
