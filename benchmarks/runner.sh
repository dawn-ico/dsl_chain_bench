#!/bin/bash

pushd build
  benches=`ls nh_diffusion_stencil_01_bench`
  for bench in $benches 
  do
    for n in 1000 10000 20000 50000 100000   
    do 
      echo "srun --partition debug --gres=gpu:1 $bench $n"
      srun --partition debug --gres=gpu:1 $bench $n
    done
  done 
popd build

# pushd build
#   benches=`ls *_bench`
#   for bench in $benches 
#   do    
#     echo "srun --partition debug --gres=gpu:1 $bench"
#     srun --partition debug --gres=gpu:1 $bench   
#   done 
# popd build