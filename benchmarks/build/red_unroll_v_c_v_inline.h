#pragma once
#include "driver-includes/defs.hpp"
#include "driver-includes/cuda_utils.hpp"
extern "C" {
void run_unroll_v_c_v_inline_from_c_host(dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *kh_smag_e, ::dawn::float_type *inv_dual_edge_length, ::dawn::float_type *theta_v, ::dawn::float_type *z_temp) ;
void run_unroll_v_c_v_inline_from_fort_host(dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *kh_smag_e, ::dawn::float_type *inv_dual_edge_length, ::dawn::float_type *theta_v, ::dawn::float_type *z_temp) ;
void run_unroll_v_c_v_inline(::dawn::float_type *kh_smag_e, ::dawn::float_type *inv_dual_edge_length, ::dawn::float_type *theta_v, ::dawn::float_type *z_temp) ;
bool verify_unroll_v_c_v_inline(const ::dawn::float_type *z_temp_dsl, const ::dawn::float_type *z_temp, const double z_temp_rel_tol, const double z_temp_abs_tol, const int iteration) ;
void run_and_verify_unroll_v_c_v_inline(::dawn::float_type *kh_smag_e, ::dawn::float_type *inv_dual_edge_length, ::dawn::float_type *theta_v, ::dawn::float_type *z_temp, ::dawn::float_type *z_temp_before, const double z_temp_rel_tol, const double z_temp_abs_tol) ;
void setup_unroll_v_c_v_inline(dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream, const int z_temp_k_size) ;
void free_unroll_v_c_v_inline() ;
}
