#!/bin/bash

source /project/g110/spack/user/tsa/spack/share/spack/setup-env.sh
spack load cmake@3.18.1

module use /apps/common/UES/sandbox/kraushm/tsa-nvhpc/easybuild/modules/all;
module load nvhpc

cmake .. -Datlas_DIR=/scratch/mroeth/spack-install/tsa/atlas/0.22.0/gcc/nv64bckunjhyj2nf6vny22hd5flrc3gj/lib64/cmake/atlas/ -Ddawn4py_DIR=/scratch/mroeth/dawn/dawn/src/ -DTOOLCHAINPATH=/scratch/mroeth/dawn/build/install/bin:/scratch/mroeth/dusk-venv/bin -DCMAKE_BUILD_TYPE=Release