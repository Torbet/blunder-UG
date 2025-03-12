#!/bin/bash

models=(dense1 dense3 dense6 conv1 conv3 conv6 convlstm transformer)
# models=(convlstm transformer)
lrs=(1e-3 5e-4 1e-4)
weight_decays=(0 1e-4)
data="processed"

# run every combination of model, lr, and weight_decay
# for transforfmer and convlstm, run with every combination of evals and times, these are either --evals or --no-evals and --times or --no-times
# --limit 10000, --num-moves 40, --save
# echo the configuration and run the training script

for model in ${models[@]}; do
    for lr in ${lrs[@]}; do
        for weight_decay in ${weight_decays[@]}; do
            if [ $model == "transformer" ] || [ $model == "convlstm" ]; then
                for evals in "--evals" "--no-evals"; do
                    for times in "--times" "--no-times"; do
                        echo "model: $model, lr: $lr, weight_decay: $weight_decay, evals: $evals, times: $times"
                        python train.py --model $model --data $data --lr $lr --weight-decay $weight_decay $evals $times --limit 10000 --num-moves 40 --save
                    done
                done
            else
                echo "model: $model, lr: $lr, weight_decay: $weight_decay"
                python train.py --model $model --data $data --lr $lr --weight-decay $weight_decay --limit 10000 --num-moves 40 --save
            fi
        done
    done
done
