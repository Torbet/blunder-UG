#!/bin/bash

train() {
    for model in ${models[@]}; do
        for lr in ${lrs[@]}; do
            for weight_decay in ${weight_decays[@]}; do
                if [ $model == "transformer" ] || [ $model == "convlstm" ]; then
                    for evals in "--evals" "--no-evals"; do
                        for times in "--times" "--no-times"; do
                            echo "model: $model, lr: $lr, weight_decay: $weight_decay, evals: $evals, times: $times, channels: $channels, data: $data, num_moves: $num_moves, engine_prob: $engine_prob, gpu: $gpu"
                            python train.py --model $model --data $data --channels $channels --lr $lr --weight-decay $weight_decay $evals $times --limit 10000 --num-moves $num_moves --engine-prob $engine_prob --gpu $gpu --save
                        done
                    done
                else
                    echo "model: $model, lr: $lr, weight_decay: $weight_decay, channels: $channels, data: $data, num_moves: $num_moves, engine_prob: $engine_prob, gpu: $gpu"
                    python train.py --model $model --data $data --channels $channels --lr $lr --weight-decay $weight_decay --limit 10000 --num-moves $num_moves --engine-prob $engine_prob --gpu $gpu --save
                fi
            done
        done
    done
}

# models=(dense1 dense3 dense6 conv1 conv3 conv6 convlstm transformer)
models=(convlstm)
data="processed"
limit=10000
lrs=(5e-4 1e-4)
weight_decays=(0 1e-4)
engine_prob=0.0
num_moves=40

gpu=1
channels=6
train

channels=12
train