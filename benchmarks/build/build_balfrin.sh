#!/bin/bash

module use $USER_ENV_ROOT/modules
module load gcc
module load cmake
module load cuda
module load python/3.10.6

cmake .. -Datlas_DIR=/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/atlas/build/install/lib64/cmake/atlas \
         -Datlas_utils_DIR=/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/AtlasUtilities/build/install/lib/cmake/atlas_utils/ \
         -Datlas_utils_LIBRARY_PATH=/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/AtlasUtilities/build/install/lib/libatlasUtilsLib.a \
         -Ddawn4py_DIR=/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/dawn/dawn/src/ \
         -DTOOLCHAINPATH=/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/dawn/build/install/bin:/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/dusk-venv/bin \
         -DCMAKE_BUILD_TYPE=Release