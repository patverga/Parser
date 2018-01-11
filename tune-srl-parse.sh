#!/bin/bash

debug="false"

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DOZAT_ROOT

OUT_LOG=$ROOT_DIR/hyperparams/tune-$timestamp
if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

echo "Writing to $OUT_LOG"

#num_gpus=108
num_gpus=24

lrs="0.04" # 0.06"
mus="0.9"
nus="0.98"
epsilons="1e-12"
warmup_steps="8000"
batch_sizes="1000"

trans_layers="4" # 3
cnn_dims="1024" # 768
num_heads="8" # 4 8"
head_sizes="64" # 128"
relu_hidden_sizes="256"
trigger_mlp_sizes="256"
trigger_pred_mlp_sizes="256"
role_mlp_sizes="256"
parse_update_proportions="1.0 0.5 0.25 0.0"
do_load_pretrained="True False"

reps="3"

# 4*2*3 = 24

load_dir="saves/trans-fast-srl-pretrain-parser-big-notrigs"



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
                                            for role_mlp_size in ${role_mlp_sizes[@]}; do
                                                for trigger_mlp_size in ${trigger_mlp_sizes[@]}; do
                                                    for trigger_pred_mlp_size in ${trigger_pred_mlp_sizes[@]}; do
                                                        for parse_update_proportion in ${parse_update_proportions[@]}; do
                                                            for load_pretrained in ${do_load_pretrained[@]}; do
                                                                for rep in `seq $reps`; do
                                                                    fname_append="$rep-$lr-$mu-$nu-$epsilon-$warmup_steps-$batch_size-$cnn_dim-$trans_layer-$num_head-$head_size-$relu_hidden_size-$role_mlp_size-$trigger_mlp_size-$trigger_pred_mlp_size-$parse_update_proportion-$load_pretrained"
                                                                    load_pretrained_str=""
                                                                    if [[ "$load_pretrained" == "True" ]]; then
                                                                        load_pretrained_str="--load_dir $load_dir --load"
                                                                    fi
                                                                    commands+=("srun --gres=gpu:1 --partition=titanx-long,m40-long --mem=12000
                                                                     python network.py \
                                                                    --config_file config/trans-conll12-bio-parse.cfg \
                                                                    --save_dir $OUT_LOG/scores-$fname_append \
                                                                    --save_every 500 \
                                                                    --train_batch_size $batch_size \
                                                                    --test_batch_size $batch_size \
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
                                                                    --trigger_mlp_size $trigger_mlp_size \
                                                                    --trigger_pred_mlp_size $trigger_pred_mlp_size \
                                                                    --role_mlp_size $role_mlp_size \
                                                                    $load_pretrained_str \
                                                                    --parse_update_proportion $parse_update_proportion \
                                                                    --svd_tree False \
                                                                    --mask_pairs True \
                                                                    --mask_roots True \
                                                                    --ensure_tree True \
                                                                    --save False \
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
