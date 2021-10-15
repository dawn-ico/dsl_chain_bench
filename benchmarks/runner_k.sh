#!/bin/bash
for k in 1 3 6 12 32 65
do
  sed -i 's/add_stencil(\${sten} [0-9]* 128 TRUE)/add_stencil(${sten} '$k' 128 TRUE)/g' CMakeLists.txt 
  sed -i 's/add_stencil(\${sten} [0-9]* 128 FALSE)/add_stencil(${sten} '$k' 128 FALSE)/g' CMakeLists.txt 
  echo $k
  pushd build
    make -j8
    echo "srun --partition debug --gres=gpu:1 nh_diffusion_stencil_01_bench"
    srun --partition debug --gres=gpu:1 nh_diffusion_stencil_01_bench
  popd
done