#!/usr/bin/env bash

filename_list=../../benchmarks/test
data_root=../../benchmarks
file_list=test
num_epochs=25

while IFS='' read -r line || [[ -n "$line" ]]; do

    bash run_single.sh $line ss_gembed

done < "$filename_list"


