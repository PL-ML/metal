#!/bin/bash

data_root=../../benchmarks
file_list=train
num_epochs=350

if [ ! -d "$data_root/${1}_meta_log" ]; then
  mkdir $data_root/${1}_meta_log
fi


python main.py -data_root $data_root \
        -file_list $file_list \
        -num_epochs $num_epochs \
        -seed 1 \
        -eps 0.85 \
        -eps_decay 0.99995 \
        -exit_on_find 0 | tee $data_root/${1}_meta_log/log

