#!/bin/bash

run_command() {
    echo "processing: $1"
    $1
    if [ $? -ne 0 ]; then
        echo "failed: $1"
        exit 1
    fi
}

run_command "python3 inductive/data_process.py"
run_command "python3 fully-inductive/data_process.py"
run_command "python3 transductive/data_process.py"
run_command "python3 ILPC/data_process.py"
run_command "python3 multi-hop/data_process.py"

echo "Finish！"
