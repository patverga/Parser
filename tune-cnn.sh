#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

#num_gpus=120
num_gpus=36

lrs="0.04" # 0.06"
mus="0.9"
nus="0.98"
epsilons="1e-12"
warmup_steps="8000"
batch_sizes="5000"

trans_layers="1 2 4" # 3
num_heads="8" #4 8"
head_sizes="64"
relu_hidden_sizes="256"

cnn2d_layers="0"
cnn_dim_2ds="0"

cnn_layers="1 2 4"
cnn_dims="384 512 768 1024"
#cnn_layers="4 6 8 10"
#cnn_dims="384 512 784 1024"

num_blocks="1 2 4"

pairs_penalties="0.0"
roots_penalties="0.0"
svd_penalties="0.0"

reps="2"

# 3*3*4*3*2

# array to hold all the commands we'll distribute
declare -a commands

for lr in ${lrs[@]}; do
    for mu in ${mus[@]}; do
        for nu in ${nus[@]}; do
            for epsilon in ${epsilons[@]}; do
                for warmup_steps in ${warmup_steps[@]}; do
                    for cnn_dim in ${cnn_dims[@]}; do
                        for trans_layer in ${trans_layers[@]}; do
                            for num_head in ${num_heads[@]}; do
                                for head_size in ${head_sizes[@]}; do
                                    for relu_hidden_size in ${relu_hidden_sizes[@]}; do
                                        for batch_size in ${batch_sizes[@]}; do
                                            for pairs_penalty in ${pairs_penalties[@]}; do
                                                for roots_penalty in ${roots_penalties[@]}; do
                                                    for svd_penalty in ${svd_penalties[@]}; do
                                                        for cnn2d_layer in ${cnn2d_layers[@]}; do
                                                            for cnn_dim_2d in ${cnn_dim_2ds[@]}; do
                                                                for cnn_layer in ${cnn_layers[@]}; do
                                                                    for num_block in ${num_blocks[@]}; do
                                                                        for rep in `seq $reps`; do
                                                                            fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size-$num_block-$cnn_layer-$cnn_dim-$trans_layer-$num_head-$head_size-$relu_hidden_size-$cnn2d_layer-$cnn_dim_2d"
                                                                            commands+=("srun --gres=gpu:1 --partition=m40-short,m40-long --time=04:00:00 python network.py \
                                                                            --config_file config/myconf.cfg \
                                                                            --save_dir $OUT_LOG/scores-$fname_append \
                                                                            --save_every 500 \
                                                                            --train_iters 100000 \
                                                                            --train_batch_size $batch_size \
                                                                            --warmup_steps $warmup_steps \
                                                                            --learning_rate $lr \
                                                                            --cnn_dim $cnn_dim \
                                                                            --n_recur $trans_layer \
                                                                            --num_heads $num_head \
                                                                            --head_size $head_size \
                                                                            --relu_hidden_size $relu_hidden_size \
                                                                            --mu $mu \
                                                                            --nu $nu \
                                                                            --epsilon $epsilon \
                                                                            --pairs_penalty $pairs_penalty \
                                                                            --roots_penalty $roots_penalty \
                                                                            --svd_penalty $svd_penalty \
                                                                            --svd_tree True \
                                                                            --mask_pairs True \
                                                                            --mask_roots True \
                                                                            --ensure_tree False \
                                                                            --save False \
                                                                            --cnn_dim_2d $cnn_dim_2d \
                                                                            --cnn2d_layers $cnn2d_layer \
                                                                            --cnn_layers $cnn_layer \
                                                                            --num_blocks $num_block \
                                                                            --dist_model transformer \
                                                                            &> $OUT_LOG/train-$fname_append.log")
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
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
if [[ $num_jobs -lt $num_gpus ]]; then
    jobs_per_gpu=1
    num_gpus=$num_jobs
else
    jobs_per_gpu=$((num_jobs / num_gpus))
fi
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
