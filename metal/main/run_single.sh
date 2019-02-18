#!/bin/bash

data_root=../../benchmarks
file_list=all
num_epochs=350

if [ ! -d "$data_root/${2}_single_log" ]; then
  mkdir $data_root/${2}_single_log
fi

python main.py -data_root $data_root \
-file_list $file_list \
-num_epochs $num_epochs \
-single_sample $1  \
-seed 1 \
-exit_on_find 1 | tee $data_root/${2}_single_log/${1}-log


