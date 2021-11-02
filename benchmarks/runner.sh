#!/bin/bash
for k in 1 3 6 12 32 65
do
  sed -i 's/add_stencil(\${sten} [0-9]* 128 TRUE)/add_stencil(${sten} '$k' 128 TRUE)/g' CMakeLists.txt 
  sed -i 's/add_stencil(\${sten} [0-9]* 128 FALSE)/add_stencil(${sten} '$k' 128 FALSE)/g' CMakeLists.txt 
  pushd build
  make -j8
  benches=`ls *_bench`
  for bench in $benches
  do  
    for n in 1000 10000 20000 50000 100000       
    do 
      # echo "srun --partition debug --gres=gpu:1 $bench $n"
      # srun --partition debug --gres=gpu:1 $bench $n
      rt=$(srun --partition debug --gres=gpu:1 $bench $n | awk '{print$5}')
      echo $bench $k $n $rt
      echo $bench $k $n $rt >> coarsening.txt
    done
  done 
  popd
done  

# pushd build
#   benches=`ls *_bench`
#   for bench in $benches
#   do
#   echo $bench 
#     for n in 1000 10000 20000 50000 100000       
#     do 
#       # echo "srun --partition debug --gres=gpu:1 $bench $n"
#       # srun --partition debug --gres=gpu:1 $bench $n
#       # mem=$(srun --partition debug --gres=gpu:1 ncu -k '.*stencil.*' $bench $n | grep 'L2' | awk '{print$5}')
#       # echo $n $mem
#       rt=$(srun --partition debug --gres=gpu:1 $bench $n | awk '{print$5}')
#       echo $bench $n $rt
#       echo $bench $n $rt >> runtimes.txt
#     done
#   done 
# popd build

# pushd build
#   benches=`ls *_bench`
#   for bench in $benches 
#   do    
#     echo "srun --partition debug --gres=gpu:1 $bench"
#     srun --partition debug --gres=gpu:1 $bench   
#   done 
# popd build