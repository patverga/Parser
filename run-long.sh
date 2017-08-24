#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

num_gpus=30

lrs="0.04 0.06"
mus="0.9"
nus="0.98"
epsilons="1e-12"
warmup_steps="4000 2000 8000"
batch_sizes="5000"

trans_layers="3 4 5"
cnn_dims="768 1048"
num_heads="4 6 8"
head_sizes="64 128"
relu_hidden_sizes="256"

reps="3"

# 3*3*2*3*2*3

params="
0.04-0.9-0.98-1e-12-4000-5000-1048-4-6-64
0.04-0.9-0.98-1e-12-4000-5000-1048-4-4-128
0.06-0.9-0.98-1e-12-8000-5000-1048-3-8-64
0.04-0.9-0.98-1e-12-8000-5000-1048-4-6-64
0.04-0.9-0.98-1e-12-8000-5000-1048-3-8-64
0.06-0.9-0.98-1e-12-8000-5000-1048-4-6-64
"

relu_hidden_sizes="256 512 768 1024"

# array to hold all the commands we'll distribute
declare -a commands

for p in ${params[@]}; do
    cm=`echo ${p} | sed 's/1e-12/.000000000001/g' | \
    awk -F'-' '{ printf \
        "--learning_rate %s \
        \n--mu %s \
        \n--nu %s \
        \n--epsilon %s \
        \n--warmup_steps %s \
        \n--train_batch_size %s \
        \n--cnn_dim %s \
        \n--n_recur %s \
        \n--num_heads %s \
        \n--head_size %s\n", \
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10}'`
    echo $cm
    for relu_hidden_size in ${relu_hidden_sizes[@]}; do
        fname_append="$p-$relu_hidden_size"
        commands+=("srun --gres=gpu:1 --partition=titanx-long python network.py \
        --config_file config/myconf.cfg \
        --save_dir $OUT_LOG/scores-$fname_append \
        --save_every 500 \
        --train_iters 500000 \
        --relu_hidden_size $relu_hidden_size \
        $cm \
        &> $OUT_LOG/train-$fname_append.log")
    done
done

# now distribute them to the gpus
num_jobs=${#commands[@]}
if [ $num_jobs -lt $num_gpus ]; then
    num_gpus=${#commands[@]}
fi
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
