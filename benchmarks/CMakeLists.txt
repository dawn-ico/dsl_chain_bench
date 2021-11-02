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
    set(dawn_args --default-opt --pass-groups set-loop-order,temporary-inlining)
  else()    
    set(dawn_args --default-opt --pass-groups set-loop-order)
  endif()  

  add_custom_target(${stencil}_sir   
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONHOME=${PYTHONHOME} PYTHONPATH=${PYTHONPATH} PATH="${TOOLCHAINPATH}"
      dusk-front ${CMAKE_CURRENT_SOURCE_DIR}/${stencil}.py > ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.sir
  ) 
  add_custom_target(${stencil}_iir
    DEPENDS ${stencil}_sir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PATH="${TOOLCHAINPATH}"
      dawn-opt ${dawn_args} -o ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.iir ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.sir
  )
  add_custom_command(
    OUTPUT ${stencil}.cpp
    DEPENDS ${stencil}_iir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PATH="${TOOLCHAINPATH}"
      dawn-codegen -b cuda-ico
      -o ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.cpp      
      --output-c-header ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.h
      --block-size ${block_size}
      --levels-per-thread ${levels_per_thread}
      ${CMAKE_CURRENT_BINARY_DIR}/${stencil}.iir
  )    
endfunction()

file(GLOB stencils_py CONFIGURE_DEPENDS "*.py")
set(stencils)
foreach (sten ${stencils_py})
    get_filename_component(cur_sten ${sten} NAME_WE)     
    list(APPEND stencils ${cur_sten})
endforeach()

foreach (sten ${stencils})
  if (${sten} MATCHES ".*inline")   
    add_stencil(${sten} 1 128 TRUE)
  else()   
    add_stencil(${sten} 1 128 TRUE)
  endif()
endforeach()

function (compile_benchmark sten_full_path)   
  get_filename_component(sten_bench ${sten_full_path} NAME_WE) 
  string(REPLACE "_bench" "" sten ${sten_bench})    
  
  set(sten_inline_full_path ${CMAKE_BINARY_DIR}/${sten}_inline.cpp)
  set(sten_unroll_full_path ${CMAKE_BINARY_DIR}/${sten}_unroll.cpp) 

  set_source_files_properties(
    ${sten_inline_full_path}   
    ${sten_unroll_full_path}    
    ${dawn4py_DIR}/driver-includes/to_vtk.cpp
    ${dawn4py_DIR}/driver-includes/cuda_utils.cpp
    ${dawn4py_DIR}/driver-includes/cuda_verify.cpp
    PROPERTIES LANGUAGE CUDA)    
  
  add_library(${sten} STATIC
    ${sten_inline_full_path}    
    ${sten_unroll_full_path}
    ${dawn4py_DIR}/driver-includes/to_vtk.cpp
    ${dawn4py_DIR}/driver-includes/cuda_utils.cpp
    ${dawn4py_DIR}/driver-includes/cuda_verify.cpp
  )    
  
  add_executable(${sten_bench}
    ${sten_full_path}
    ${CMAKE_CURRENT_LIST_DIR}/utils/atlasToGlobalGpuTriMesh.cpp
  )
  
  target_include_directories(${sten} PUBLIC ${dawn4py_DIR})
  target_compile_options(${sten} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>) 
  # target_compile_options(${sten} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ptx>)
  # target_compile_options(${sten} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-src-in-ptx>)

  target_include_directories(${sten_bench} PRIVATE ${dawn4py_DIR})
  target_include_directories(${sten_bench} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/utils)
  target_include_directories(${sten_bench} PRIVATE ${CMAKE_BINARY_DIR})
  target_include_directories(${sten_bench} PRIVATE ${atlas_utils_INCLUDE_DIRS})    
  target_link_libraries(${sten_bench} PRIVATE atlas)
  target_link_libraries(${sten_bench} PRIVATE thrustUtils)
  target_link_libraries(${sten_bench} PRIVATE ${sten})
  target_link_libraries(${sten_bench} PRIVATE atlasUtilsLib)   
endfunction()

file(GLOB benches CONFIGURE_DEPENDS "*_bench.cpp")
foreach(sten ${benches})    
  compile_benchmark(${sten}) 
endforeach()

configure_file(resources/grid.nc ${PROJECT_BINARY_DIR}/grid.nc COPYONLY)
