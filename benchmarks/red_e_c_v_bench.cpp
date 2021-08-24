#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

namespace inlined {
  #include "red_e_c_v_inline.h"
}

namespace sequential {
  #include "red_e_c_v_sequential.h"
}


template<typename... Args>
double run_and_time(void (*fun) (Args... args), Args... args) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  fun(args...);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  return milliseconds;
}

int main() {
  atlas::Mesh mesh = AtlasMeshFromNetCDFComplete("grid.nc");
  dawn::GlobalGpuTriMesh gpu_tri_mesh = atlasToGlobalGpuTriMesh(mesh);
  const int num_lev = 80;

  const size_t in_size = mesh.nodes().size()*num_lev;
  const size_t out_size = mesh.edges().size()*num_lev;
  double *in_field_nested, *out_field_nested;
  double *in_field_sequential, *out_field_sequential;
  
  cudaMalloc((void**)&in_field_nested, in_size*sizeof(double));
  cudaMalloc((void**)&out_field_nested, out_size*sizeof(double));
  cudaMalloc((void**)&in_field_sequential, in_size*sizeof(double));
  cudaMalloc((void**)&out_field_sequential, out_size*sizeof(double));

  fill_random(in_field_nested, in_size);
  cudaMemcpy(in_field_sequential, in_field_nested, in_size*sizeof(double), cudaMemcpyDeviceToDevice);

  cudaStream_t stream;
  inlined::setup_red_e_c_v(&gpu_tri_mesh, num_lev, stream);
  sequential::setup_red_e_c_v(&gpu_tri_mesh, num_lev, stream);

  double time_nested = run_and_time(inlined::run_red_e_c_v, in_field_nested, out_field_nested);
  double time_sequential = run_and_time(sequential::run_red_e_c_v, in_field_sequential, out_field_sequential);

  printf("E > C > V: seq %e nest %e\n", time_nested, time_sequential);

  bool valid_result = verify(out_field_nested, out_field_sequential, out_size, 1e-12);
  if (!valid_result) {
    printf("[FAIL] Failed Verification!");
  }

  return 0;
}