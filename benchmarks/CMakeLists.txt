cmake_minimum_required(VERSION 3.18)

project(DSL-Benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# CUDA
if(DEFINED ENV{CUDAARCHS})
  # respect environment variable `CUDAARCHS` which was introduced in cmake 3.20
  set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architectures")
else()
  # this needs to happen before `enable_language(CUDA)`, because
  # `enable_language(CUDA)` sets `CMAKE_CUDA_ARCHITECTURES`
  set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA architectures")
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_EXTENSIONS OFF)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT EXISTS "${dawn4py_DIR}")
  message(FATAL_ERROR "Cannot find include path (dawn4py).")
endif()


### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

if(DEFINED PYTHONHOME)
  set(PYTHONHOME "${PYTHONHOME}" CACHE INTERNAL "PYTHONHOME of python environment to use")
else()
  set(PYTHONHOME "$ENV{PYTHONHOME}" CACHE INTERNAL "PYTHONHOME of python environment to use")
endif()

if(DEFINED PYTHONPATH)
  set(PYTHONPATH "${PYTHONPATH}" CACHE INTERNAL "PYTHONPATH of python environment to use")
else()
  set(PYTHONPATH "$ENV{PYTHONPATH}" CACHE INTERNAL "PYTHONPATH of python environment to use")
endif()

if(DEFINED TOOLCHAINPATH)
  set(TOOLCHAINPATH "${TOOLCHAINPATH}" CACHE INTERNAL "Path variable with directories to search tools in")
else()
  set(TOOLCHAINPATH "$ENV{PATH}" CACHE INTERNAL "Path variable with directories to search tools in")
endif()

set(dusk_stencils "")

function (add_stencil stencil levels_per_thread block_size)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.sir
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${stencil}.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONHOME=${PYTHONHOME} PYTHONPATH=${PYTHONPATH} PATH="${TOOLCHAINPATH}"
      dusk-front ${CMAKE_CURRENT_SOURCE_DIR}/${stencil}.py > ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.sir
  )
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.iir
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.sir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PATH="${TOOLCHAINPATH}"
      dawn-opt --default-opt --pass-groups set-loop-order -o ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.iir ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.sir
  )
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.cpp ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.f90 ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.h
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.iir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PATH="${TOOLCHAINPATH}"
      dawn-codegen -b cuda-ico
      -o ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.cpp
      --output-f90-interface ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.f90
      --output-c-header ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.h
      --block-size ${block_size}
      --levels-per-thread ${levels_per_thread}
      ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.iir
  )
  set(dusk_stencils ${dusk_stencils} ${stencil} PARENT_SCOPE)

  set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.f90
    PROPERTIES Fortran_PREPROCESS ON
  )
endfunction()

add_stencil("red_e_c_v" 128 1)

list(TRANSFORM dusk_stencils PREPEND ${CMAKE_CURRENT_BINARY_DIR}/
     OUTPUT_VARIABLE dusk_stencils_paths)

set(dusk_stencil_cpp_sources ${dusk_stencils_paths})

list(TRANSFORM dusk_stencil_cpp_sources APPEND .cpp)

add_executable(sten 
  ${dusk_stencil_cpp_sources}    
  ${dawn4py_DIR}/driver-includes/to_vtk.cpp
  ${dawn4py_DIR}/driver-includes/cuda_utils.cpp
  ${dawn4py_DIR}/driver-includes/cuda_verify.cpp
)
