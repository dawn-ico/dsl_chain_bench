#!/bin/bash

source /project/g110/spack/user/tsa/spack/share/spack/setup-env.sh
spack load cmake@3.18.1

module use /apps/common/UES/sandbox/kraushm/tsa-nvhpc/easybuild/modules/all;
module load nvhpc

cmake .. -Ddawn4py_DIR=/scratch/mroeth/dawn/dawn/src/ -DTOOLCHAINPATH=/scratch/mroeth/dawn/build/install/bin:/scratch/mroeth/dusk-venv/bin -DCMAKE_BUILD_TYPE=Release