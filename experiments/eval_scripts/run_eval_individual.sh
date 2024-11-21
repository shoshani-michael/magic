#!/bin/bash

export MODEL=$1
export DATASET=$2
keywords=($MODEL $DATASET)

folder_path="/public/home/hongy/jiahli/llm-attacks/experiments/results"
matching_files=()
for filename in "$folder_path"/*; do
    match=true
    for keyword in "${keywords[@]}"; do
        if ! [[ "$filename" == *"$keyword"* ]]; then
            match=false
            break
        fi
    done
    if [ "$match" = true ]; then
        matching_files+=("../results/$(basename "$filename")")
    fi
done

for LOG in "${matching_files[@]}"
do
    python -u ../evaluate_individual.py \
        --config=../configs/transfer.py \
        --config.train_data="../../data/advbench/harmful_behaviors.csv" \
        --config.logfile="$LOG" \
        --config.n_train_data=1 \
        --config.n_test_data=0 \
done
