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

find_package(atlas CONFIG REQUIRED)
find_package(atlas_utils CONFIG REQUIRED) 

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

add_subdirectory(utils)

function (add_stencil stencil levels_per_thread block_size inline)

  if (inline)
    set(sten_name ${stencil}_inline)
    set(dawn_args --default-opt --pass-groups set-loop-order,temporary-inlining)
  else()
    set(sten_name ${stencil}_sequential)
    set(dawn_args --default-opt --pass-groups set-loop-order)
  endif()  

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.sir
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${stencil}.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONHOME=${PYTHONHOME} PYTHONPATH=${PYTHONPATH} PATH="${TOOLCHAINPATH}"
      dusk-front ${CMAKE_CURRENT_SOURCE_DIR}/${stencil}.py > ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.sir
  )
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.iir
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.sir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PATH="${TOOLCHAINPATH}"
      dawn-opt ${dawn_args} -o ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.iir ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.sir
  )
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.cpp ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.f90 ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.h
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.iir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PATH="${TOOLCHAINPATH}"
      dawn-codegen -b cuda-ico
      -o ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.cpp      
      --output-c-header ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.h
      --block-size ${block_size}
      --levels-per-thread ${levels_per_thread}
      ${CMAKE_CURRENT_BINARY_DIR}/${sten_name}.iir
  )    
endfunction()

file(GLOB stencils_py CONFIGURE_DEPENDS "*.py")
set(stencils)
foreach (sten ${stencils_py})
    get_filename_component(cur_sten ${sten} NAME_WE)     
    list(APPEND stencils ${cur_sten})
endforeach()

foreach (sten ${stencils})
  add_stencil(${sten} 128 1 TRUE)
  add_stencil(${sten} 128 1 FALSE)
endforeach()

foreach(sten ${stencils})    
    set(sten_bench ${CMAKE_CURRENT_LIST_DIR}/${sten}_bench.cpp)
    set(sten_sequential ${CMAKE_CURRENT_BINARY_DIR}/${sten}_sequential.cpp)
    set(sten_inline ${CMAKE_CURRENT_BINARY_DIR}/${sten}_inline.cpp)    

    add_library(${sten} STATIC
      ${sten_sequential}
      ${sten_inline}
      ${dawn4py_DIR}/driver-includes/to_vtk.cpp
      ${dawn4py_DIR}/driver-includes/cuda_utils.cpp
      ${dawn4py_DIR}/driver-includes/cuda_verify.cpp)

    set_source_files_properties(
      ${sten_sequential}
      ${sten_inline}   
      ${dawn4py_DIR}/driver-includes/to_vtk.cpp
      ${dawn4py_DIR}/driver-includes/cuda_utils.cpp
      ${dawn4py_DIR}/driver-includes/cuda_verify.cpp
      PROPERTIES LANGUAGE CUDA)    
      
    target_include_directories(${sten} PUBLIC ${dawn4py_DIR})

    add_executable(${sten}_bench
      ${sten_bench}
      ${CMAKE_CURRENT_LIST_DIR}/utils/atlasToGlobalGpuTriMesh.cpp
    )
    target_include_directories(${sten}_bench PRIVATE ${dawn4py_DIR})
    target_include_directories(${sten}_bench PRIVATE ${CMAKE_CURRENT_LIST_DIR}/utils)
    target_include_directories(${sten}_bench PRIVATE ${CMAKE_BINARY_DIR})
    target_include_directories(${sten}_bench PRIVATE ${atlas_utils_INCLUDE_DIRS})    
    target_link_libraries(${sten}_bench PRIVATE atlas)
    target_link_libraries(${sten}_bench PRIVATE thrustUtils)
    target_link_libraries(${sten}_bench PRIVATE ${sten})
    target_link_libraries(${sten}_bench PRIVATE atlasUtilsLib)    
endforeach()

configure_file(resources/grid.nc ${PROJECT_BINARY_DIR}/grid.nc COPYONLY)