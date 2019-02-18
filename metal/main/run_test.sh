#!/usr/bin/env bash

data_root=../../benchmarks
file_list=test
num_epochs=350

while IFS='' read -r line || [[ -n "$line" ]]; do

    python main.py -data_root $data_root \
        -file_list $file_list \
        -num_epochs $num_epochs \
        -single_sample $line  \
        -tune_test 1 \
        -seed 1 \
        -exit_on_find 1

done < "$filename_list"

