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
num_gpus=96

lrs="0.04 0.1 0.01" # 0.06"
mus="0.9"
nus="0.98 0.9"
epsilons="1e-12"
warmup_steps="8000"
batch_sizes="5000"

lstm_layers="1 2 4 6"

cnn2d_layers="0"
cnn_dim_2ds="0"

cnn_layers="0"
cnn_dims="0"
#cnn_layers="4 6 8 10"
#cnn_dims="384 512 784 1024"

num_blocks="1"

pairs_penalties="0.0"
roots_penalties="0.0"
svd_penalties="0.0"

residuals="True False"

reps="2"

# 4*2*2*2*3

# array to hold all the commands we'll distribute
declare -a commands

for lr in ${lrs[@]}; do
    for mu in ${mus[@]}; do
        for nu in ${nus[@]}; do
            for epsilon in ${epsilons[@]}; do
                for warmup_step in ${warmup_steps[@]}; do
                    for cnn_dim in ${cnn_dims[@]}; do
                        for lstm_layer in ${lstm_layers[@]}; do
                            for batch_size in ${batch_sizes[@]}; do
                                for pairs_penalty in ${pairs_penalties[@]}; do
                                    for roots_penalty in ${roots_penalties[@]}; do
                                        for svd_penalty in ${svd_penalties[@]}; do
                                            for cnn2d_layer in ${cnn2d_layers[@]}; do
                                                for cnn_dim_2d in ${cnn_dim_2ds[@]}; do
                                                    for cnn_layer in ${cnn_layers[@]}; do
                                                        for num_block in ${num_blocks[@]}; do
                                                            for residual in ${residuals[@]}; do
                                                                for rep in `seq $reps`; do
                                                                    fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_step-$batch_size-$num_block-$cnn_layer-$cnn_dim-$lstm_layer-$residual-$cnn2d_layer-$cnn_dim_2d"
                                                                    commands+=("srun --gres=gpu:1 --partition=titanx-short python network.py \
                                                                    --config_file config/myconf.cfg \
                                                                    --save_dir $OUT_LOG/scores-$fname_append \
                                                                    --save_every 500 \
                                                                    --train_iters 100000 \
                                                                    --train_batch_size $batch_size \
                                                                    --warmup_steps $warmup_step \
                                                                    --learning_rate $lr \
                                                                    --cnn_dim $cnn_dim \
                                                                    --n_recur $lstm_layer \
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
                                                                    --dist_model bilstm \
                                                                    --lstm_residual $residual \
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
