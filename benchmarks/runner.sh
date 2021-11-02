#!/bin/bash

pushd build
  benches=`ls red_unroll_*_bench`
  for bench in $benches 
  do
    echo "srun --partition debug --gres=gpu:1 $bench"
    rt=$(srun --partition debug --gres=gpu:1 $bench)
    echo $rt >> unrolled_times.txt
  done 
popd build