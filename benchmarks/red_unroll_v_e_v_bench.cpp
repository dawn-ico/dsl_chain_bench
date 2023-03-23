#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"

#include <numeric>

#include "red_unroll_v_e_v_inline.h"
#include "red_unroll_v_e_v_unroll.h"

template <typename... Args>
double run_and_time(void (*fun)(Args... args), Args... args)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, (cudaStream_t)0);
  fun(args...);
  cudaEventRecord(stop, (cudaStream_t)0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds;
}

int main()
{
  atlas::Mesh mesh = *AtlasMeshFromNetCDFComplete("grid.nc");
  dawn::GlobalGpuTriMesh gpu_tri_mesh = atlasToGlobalGpuTriMesh(mesh);
  const int num_lev = 65;
  const int num_runs = 1e5;

  const size_t end_size = mesh.nodes().size() * num_lev;
  const size_t mid_size = mesh.edges().size() * num_lev;
  const size_t start_size = mesh.nodes().size() * num_lev;

  double *theta_v;
  double *kh_smag_e;
  double *inv_dual_edge_length;
  double *outF_inline;
  double *outF_unroll;

  cudaMalloc((void **)&theta_v, end_size * sizeof(double));
  cudaMalloc((void **)&kh_smag_e, mid_size * sizeof(double));
  cudaMalloc((void **)&inv_dual_edge_length, mid_size * sizeof(double));
  cudaMalloc((void **)&outF_inline, start_size * sizeof(double));
  cudaMalloc((void **)&outF_unroll, start_size * sizeof(double));

  fill_random(theta_v, end_size);
  fill_random(kh_smag_e, mid_size);
  fill_random(inv_dual_edge_length, mid_size);

  // gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0, 3160);
  // gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Nudging, 0, 410);
  // gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Nudging, 0, 410);

  // gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Halo, 0, 20339);
  // gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Halo, 0, 30714);
  // gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Halo, 0, 10375);

  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0, 3160);
  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Nudging, 0, 5134);
  gpu_tri_mesh.set_splitter_index_lower(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Nudging, 0, 1209);

  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Halo, 0, 20339);
  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Edges, dawn::UnstructuredSubdomain::Halo, 0, 30714);
  gpu_tri_mesh.set_splitter_index_upper(dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Halo, 0, 10375);

  setup_unroll_v_e_v_inline(&gpu_tri_mesh, num_lev, (cudaStream_t)0);
  setup_unroll_v_e_v_unroll(&gpu_tri_mesh, num_lev, (cudaStream_t)0, num_lev);

  std::vector<double> times_inlined;
  std::vector<double> times_unrolled;
  for (int i = 0; i < num_runs; i++)
  {
    double time_unrolled = run_and_time(run_unroll_v_e_v_unroll, kh_smag_e, inv_dual_edge_length, theta_v, outF_unroll);
    double time_inlined = run_and_time(run_unroll_v_e_v_inline, kh_smag_e, inv_dual_edge_length, theta_v, outF_inline);
    times_inlined.push_back(time_inlined);
    times_unrolled.push_back(time_unrolled);
  }
  auto avg = [](const std::vector<double> &in)
  { return std::accumulate(in.begin(), in.end(), 0.0) / in.size(); };
  auto sd = [&avg](const std::vector<double> &in)
  {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean); };
  printf("V > E > V: inl %e %e unroll %e %e\n", avg(times_inlined), sd(times_inlined), avg(times_unrolled), sd(times_unrolled));

  bool valid_result = verify(outF_inline, outF_unroll, start_size, 1e-12);
  if (!valid_result)
  {
    printf("[FAIL] Failed Verification!\n");
  }

  return 0;
}
