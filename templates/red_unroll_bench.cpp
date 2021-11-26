#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"

#include <numeric>

#include "red_unroll_{CHAIN_LETTERS}_inline.h"
#include "red_unroll_{CHAIN_LETTERS}_sequential.h"

template<typename... Args>
double run_and_time(void (*fun) (Args... args), Args... args) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start, (cudaStream_t) 0);
  fun(args...);
  cudaEventRecord(stop, (cudaStream_t) 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);   
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return milliseconds;
}

int main() {
  atlas::Mesh mesh = *AtlasMeshFromNetCDFComplete("grid.nc");
  dawn::GlobalGpuTriMesh gpu_tri_mesh = atlasToGlobalGpuTriMesh(mesh);
  const int num_lev = 65;
  const int num_runs = 100000;

  const size_t in_size = {CHAIN_2_MESH}.size()*num_lev; 
  const size_t out_size = {CHAIN_0_MESH}.size()*num_lev;
  const size_t mid_size_flat = {CHAIN_1_MESH}.size();
  const size_t mid_size = {CHAIN_1_MESH}.size()*num_lev;
  
  double *in_field_inlined, *out_field_inlined;
  double *in_field_sequential, *out_field_sequential; 

  double *scale0, *scale1; 

  cudaMalloc((void**)&in_field_inlined, in_size*sizeof(double));
  cudaMalloc((void**)&out_field_inlined, out_size*sizeof(double));
  cudaMalloc((void**)&in_field_sequential, in_size*sizeof(double));
  cudaMalloc((void**)&out_field_sequential, out_size*sizeof(double));

  cudaMalloc((void**)&scale0, mid_size*sizeof(double));
  cudaMalloc((void**)&scale1, mid_size_flat*sizeof(double));  

  fill_random(in_field_inlined, in_size);

  fill_random(scale0, mid_size);
  fill_random(scale1, mid_size_flat);  

  cudaMemcpy(in_field_sequential, in_field_inlined, in_size*sizeof(double), cudaMemcpyDeviceToDevice);
  
  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0, 3160);  
  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Nudging, 0, 5134);  
  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Nudging, 0, 1209);

  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Halo, 0, 20339);
  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Halo, 0, 30714);
  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Halo, 0, 10375);
  
  setup_unroll_{CHAIN_LETTERS}_inline(&gpu_tri_mesh, num_lev, (cudaStream_t) 0, num_lev);
  setup_unroll_{CHAIN_LETTERS}_sequential(&gpu_tri_mesh, num_lev, (cudaStream_t) 0, num_lev);    

  std::vector<double> times_inlined, times_sequential;
  for (int i = 0; i < num_runs; i++) {
    double time_inlined = run_and_time(run_unroll_{CHAIN_LETTERS}_inline, scale0, scale1, in_field_inlined, out_field_inlined);
    double time_sequential = run_and_time(run_unroll_{CHAIN_LETTERS}_sequential, scale0, scale1, in_field_sequential, out_field_sequential);
    times_inlined.push_back(time_inlined);
    times_sequential.push_back(time_sequential);
  }
  auto avg = [] (const std::vector<double>& in) {return std::accumulate( in.begin(), in.end(), 0.0) / in.size();};
  auto sd = [&avg] (const std::vector<double>& in) {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean);};
  printf("{CHAIN_PRINT}: seq %e %e inl %e %e\n", avg(times_sequential), sd(times_sequential), avg(times_inlined), sd(times_inlined));

  bool valid_result = verify(out_field_inlined, out_field_sequential, out_size, 1e-12);
  if (!valid_result) {
    printf("[FAIL] Failed Verification!\n");
  }

  return 0;
}