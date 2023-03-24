# Installing dsl_chain_bench on Balfrin

`dsl_chain_bench` currently has quite a few pre-requisites. Once by way of dusk and dawn, but then by itself because it requires `AtlasUtilities`, which in turn requires `atlas` and `eckit`. Should `dsl_chain_bench` be maintained for whatever reason beyond repeating the benchmarks already present on balfrin, it is recommended that the dependence on `AtlasUtilities` is removed. This can be achieved by writing a new utility function that leverages `netcdf-cxx4` to read in the neighbor lists directly. This hypothetical utility may be inspired by `icon_setup.cpp` [here](https://github.com/C2SM/icon-exclaim/blob/icon-dsl/dsl/icon_setup.cpp). 

It's probably best you make a new empty folder on your scratch, possibly called `dsl_synthetic_bench` or so. This folder will be refered to as `<your/scratch/dsl_synthetic_bench>` throghout this guide. Also, make sure to source a spack-c2sm installation (`source <path/to/spack_folder/spack-c2sm/setup/setup-env.sh`). 

Install **python 3.8** using spack (dusk is only compatible with python3.8):

```
spack install python@3.8.13%gcc@11.3.0
```

Unfortunately, llvm (needed for dawn) can not be installed using spack due to a bizare python version problem. Thus, install **llvm 10** manually instead

```
cd <your/scratch/dsl_synthetic_bench>
#it is extremely important to install out of the build folder for some reason
mkdir llvm10 #

module load gcc/11.3.0
module load cmake/3.24.2-gcc
module load python/3.10.6

git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout refs/tags/llvmorg-10.0.0
mkdir build && cd build && mkdir install
cmake ../llvm -DLLVM_ENABLE_PROJECTS=clang -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_TERMINFO=OFF -DCMAKE_INSTALL_PREFIX=<your/scratch/dsl_synthetic_bench>/llvm10
#gcc 11.3 is a bit strict, let's hotfix a small issue
printf '%s\n%s\n' "#include <limits>" "$(cat ../llvm/utils/benchmark/src/benchmark_register.h)" > ../llvm/utils/benchmark/src/benchmark_register.h
srun -p postproc -uc 128 make -j 128
make -j8 install
```

Install **dusk** & **dawn**

First
```
cd <your/scratch/dsl_synthetic_bench>
```

It's probably best you store and execute this as a script. 

```
#!/bin/bash
set -e -x

module use $USER_ENV_ROOT/modules
module load gcc
module load cmake

spack load python@3.8

python -m venv dusk-venv
source dusk-venv/bin/activate
pip install --upgrade pip setuptools wheel

# Clone and build/install dawn
git clone -b experimentalInlinePass https://github.com/MeteoSwiss-APN/dawn
pushd dawn
    mkdir build
    pushd build
        mkdir install
        cmake -DLLVM_ROOT=/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/llvm10 -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DGTCLANG_BUILD_TESTING=OFF -DDAWN_BUILD_TESTING=OFF ..
        srun -p postproc -uc 36 make -j 36
        srun -p postproc -uc 8 make install -j 8
    popd
popd

# Set up dusk (branch 'horizon') and install dawn and dusk in virtualenv
git clone https://github.com/dawn-ico/dusk.git
pushd dusk
    git checkout horizon
popd

export LLVM_ROOT=/scratch/e1000/meteoswiss/scratch/mroeth/dsl_synthetic_bench/llvm10
pip install -e dawn/dawn
pip install -e dusk
deactivate
```

Install **atlas**, **eckit** and **atlas_utils** (all required for the bench)

For eckit
```
spack install eckit%gcc@11.3.0~mpi
```

Note the `~mpi`. This disables mpi support in eckit. This is important, because otherwise running the benchmarks in the final step will result in a cryptic mpi error. Disabling mpi is fine, the bench, as it exists Today, only runs single node experiments. 

For atlas
```
cd <your/scratch/dsl_synthetic_bench>
git@github.com:ecmwf/atlas.git
cd atlas
spack load eckit

# Environment --- Edit as needed
ATLAS_SRC=$(pwd)
ATLAS_BUILD=build
ATLAS_INSTALL=$(pwd)/build/install

# 1. Create the build directory:
mkdir $ATLAS_BUILD
cd $ATLAS_BUILD
mkdir install

# 2. Run CMake
ecbuild --prefix=$ATLAS_INSTALL -- $ATLAS_SRC

# 3. Compile / Install
make -j10
make install

# 4. Check installation
$ATLAS_INSTALL/bin/atlas --info
```

For atlas utils, first install netcdf-c and netcdf-cxx4 using spack
```

spack install netcdf-c%gcc@11.3.0
spack install netcdf-cxx4%gcc@11.3.0
```

And make sure to load them 
```
spack load netcdf-c@4.8.1%gcc@11.3.0
spack load netcdf-cxx4
```

Then:
```
cd <your/scratch/dsl_synthetic_bench>
git clone https://github.com/dawn-ico/AtlasUtilities
cd AtlasUtilities
mkdir build && cd build && mkdir install
cmake .. -Deckit_DIR=$(spack find --format "{prefix}" eckit | head -n1)/lib64/cmake/eckit -Datlas_DIR=<your/scratch/dsl_synthetic_bench>/atlas/build/install/lib64/cmake/atlas -Dnetcdfcxx4_DIR=$(spack find --format "{prefix}" netcdf-cxx4 | head -n1)  -DCMAKE_INSTALL_PREFIX=$(pwd)/install
make -j8
make install
```


Then, finally, clone the actual repo

```
cd <your/scratch/dsl_synthetic_bench>
git clone git@github.com:dawn-ico/dsl_chain_bench.git
```

The general idea is now to generate some benchmarks using the `generate.py` script, and subsequently build them using cmake, so 

```
cd dsl_chain_bench
python generate.py
```

Basically, `generate.py` takes the "templates" in the `templates` directory and generates benchmark files (cpp) and dusk files (py) for all 12 possible reduction sequences. A variable `NAME` at the very top of the script determines the template to be generated. 

Then do the following to build the benchmarks

```
cd benchmarks/build
module use $USER_ENV_ROOT/modules
module load gcc
module load cmake
module load cuda
module load python/3.10.6

cmake .. -Datlas_DIR=<your/scratch/dsl_synthetic_bench>/atlas/build/install/lib64/cmake/atlas \
         -Datlas_utils_DIR=<your/scratch/dsl_synthetic_bench>/AtlasUtilities/build/install/lib/cmake/atlas_utils/ \
         -Datlas_utils_LIBRARY_PATH=<your/scratch/dsl_synthetic_bench>/AtlasUtilities/build/install/lib/libatlasUtilsLib.a \
         -Ddawn4py_DIR=<your/scratch/dsl_synthetic_bench>/dawn/dawn/src/ \
         -DTOOLCHAINPATH=<your/scratch/dsl_synthetic_bench>/dawn/build/install/bin:<your/scratch/dsl_synthetic_bench>/dusk-venv/bin
```

## Running Benchmarks Using the Default (naive) Inlining Technique

You can run all benchmarks in sequence using the `runner.sh` script in the `benchmarks` folder. The runtimes are emitted into a text file called `default_times.txt` in the `build` folder. Note that some inlined stencils may fail the verification due to overcomputation in the inlined case. This could be improved by not checking the error at the domain boundaries.  

## Running Benchmarks Using more Advanced Inlining Strategies

Note: it is strongly recommended that you delete the contents of the build folder completely when switching inlining techniques

* Running the **compressed** benchmarks (using the terminology in [this slideset](https://docs.google.com/presentation/d/1XLcZt83fxTN5UalYZPhYelgHcu7zKC_l7SzwTlwyFys/edit?usp=sharing)) is a bit cumbersome. First, checkout the custom branch `compressed` by doing `git checkout compressed`
    * Run `generate.py`
    * Switch to `benchmarks/build` and `make -j8`. If you study one of the generated inlined stencils you will notice that the stencil is invalid. In particular, the weights vector employed only has a single entry. 
    * In order to correct this, and generate the final, correct stencils, an additional python script has to be ran, placed in the `benchmarks/build` folder. So run `python inject_patches_unrolled.py`. This python script will manipulate the emitted stencil code directly, and certainly does not live up to any python coding standards (and is expected to be very brittle)
    * Build again: `make -j8`
    * An run `cd .. && ./runner.sh`
* Running the **compressed and unrolled** benchmarks (using the terminology in [this slideset](https://docs.google.com/presentation/d/1XLcZt83fxTN5UalYZPhYelgHcu7zKC_l7SzwTlwyFys/edit?usp=sharing)) is quite easy. In this case, the compressed & unrolled stencils are *not* compiled using the dusk/dawn toolchain but have been hand written and checked into the  repo on a custom branch, somewhat uncanonically, directly into the build folder. So, just `git checkout unrolling` and rebuild will compile all compressed and unrolled benchmarks. The `runner.sh` script on this branch is adapted to run just the compressed and unrolled benchmarks (just 6 of the possible 12 reductions have been implemented)

## Some Notes on the Mesh the Benchmarks are Performed On

* The mesh being used is checked into the github repository [here](https://github.com/dawn-ico/dsl_chain_bench/tree/master/benchmarks/resources), even though it is quite a large binary file. Be that as it may, this is the mesh used by the experiment `mch_ch_r04b09` (without the `_dsl` suffix).
* Should the mesh be changed in the future (perhaps to the operational 1e or 2e mesh), the subdomain markers present in all `*_bench.cpp` files need to be adapted. Currently they are set to conservatively exclude the boundary regions (such that one does not need to check for missing neighbors in neighbor lists). The same files also control how many vertical levels there are used for the benchmark. 