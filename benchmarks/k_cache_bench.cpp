#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"

#include <numeric>

#include "k_cache_inline.h"
#include "k_cache_sequential.h"

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

  const size_t cell_size = mesh.cells().size()*num_lev;  

  double *ddt_w_adv_cached, *ddt_w_adv_default;
  double *z_w_con_c, *w, *coeff1_dwdz, *coeff2_dwdz;

  cudaMalloc((void**)&ddt_w_adv_cached, cell_size*sizeof(double));  
  cudaMalloc((void**)&ddt_w_adv_default, cell_size*sizeof(double));  

  cudaMalloc((void**)&z_w_con_c, cell_size*sizeof(double));  
  cudaMalloc((void**)&w, cell_size*sizeof(double));  
  cudaMalloc((void**)&coeff1_dwdz, cell_size*sizeof(double));  
  cudaMalloc((void**)&coeff2_dwdz, cell_size*sizeof(double));   

  fill_random(z_w_con_c, cell_size);
  fill_random(w, cell_size);
  fill_random(coeff1_dwdz, cell_size);
  fill_random(coeff2_dwdz, cell_size);  

  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0, 3160);
  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Nudging, 0, 5134);
  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Nudging, 0, 1209);

  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Halo, 0, 20339);
  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Halo, 0, 30714);
  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Halo, 0, 10375);

  setup_k_cache_sequential(&gpu_tri_mesh, num_lev, (cudaStream_t) 0);
  setup_k_cache_inline(&gpu_tri_mesh, num_lev, (cudaStream_t) 0);

  std::vector<double> times_inlined, times_sequential;
  for (int i = 0; i < num_runs; i++) {
    double time_inlined = run_and_time(run_k_cache_inline, z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv_cached);
    double time_sequential = run_and_time(run_k_cache_sequential, z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv_default);
    times_inlined.push_back(time_inlined);
    times_sequential.push_back(time_sequential);
  }
  auto avg = [] (const std::vector<double>& in) {return std::accumulate( in.begin(), in.end(), 0.0) / in.size();};
  auto sd = [&avg] (const std::vector<double>& in) {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean);};
  printf("cached: %e %e, default: %e %e\n", avg(times_inlined), sd(times_inlined), avg(times_sequential), sd(times_sequential));

  bool valid_result = verify(ddt_w_adv_default, ddt_w_adv_cached, cell_size, 1e-12);
  if (!valid_result) {
    printf("[FAIL] Failed Verification!\n");
  }

  return 0;
}
