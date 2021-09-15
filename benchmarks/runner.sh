#!/bin/bash

pushd build
  benches=`ls red_int_diff_*_bench`
  for bench in $benches 
  do
    echo "srun --partition debug --gres=gpu:1 $bench"
    srun --partition debug --gres=gpu:1 $bench
  done 
popd build