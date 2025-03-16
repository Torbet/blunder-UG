#!/bin/bash

evaluate() {
    for model in "${models[@]}"; do
        if [ "$model" == "transformer" ] || [ "$model" == "convlstm" ]; then
            for evals in "--evals" "--no-evals"; do
                for times in "--times" "--no-times"; do
                    echo "Evaluating model: $model, evals: $evals, times: $times, channels: $channels"
                    python evaluate_human.py --model "$model" --channels "$channels" $evals $times
                done
            done
        else
            echo "Evaluating model: $model, channels: $channels"
            python evaluate_human.py --model "$model" --channels "$channels"
        fi
    done
}

# Only the allowed models for evaluation
models=(convlstm transformer)

# Evaluate using 12-channel configuration
channels=12
evaluate