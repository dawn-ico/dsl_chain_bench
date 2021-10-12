#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"

#include <numeric>

#include "nh_diffusion_stencil_03.h"

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

int main(int argc, char* argv[]) {
  atlas::Mesh mesh;
  if (argc < 2) {
    mesh = *AtlasMeshFromNetCDFComplete("grid.nc");
  } else {
    int nperdim = sqrt(atoi(argv[1]));
    mesh = AtlasMeshRect(nperdim, nperdim);
  }
   
  dawn::GlobalGpuTriMesh gpu_tri_mesh = atlasToGlobalGpuTriMesh(mesh);
  const int num_lev = 65;
  const int num_runs = 100000;

  const size_t num_edges = gpu_tri_mesh.NumEdges;
  const size_t num_cells = gpu_tri_mesh.NumCells;
  const size_t num_vertices = gpu_tri_mesh.NumVertices;
  const size_t cells_per_edge = 3;
  
  double *kh_smag_ec;
  double *vn;
  double *e_bln_c_s;
  double *geofac_div;
  double *diff_multfac_smag;
  double *kh_c;
  double *div;

  cudaMalloc((void**) &kh_smag_ec, num_edges*num_lev*sizeof(double));
  cudaMalloc((void**) &vn, num_edges*num_lev*sizeof(double));
  cudaMalloc((void**) &e_bln_c_s, num_cells*cells_per_edge*sizeof(double));
  cudaMalloc((void**) &geofac_div, num_cells*cells_per_edge*sizeof(double));
  cudaMalloc((void**) &diff_multfac_smag, num_lev*sizeof(double));
  cudaMalloc((void**) &kh_c, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &div, num_lev*num_cells*sizeof(double));

  fill_random(kh_smag_ec, num_edges*num_lev);
  fill_random(vn, num_edges*num_lev);
  fill_random(e_bln_c_s, num_cells*cells_per_edge);
  fill_random(geofac_div, num_cells*cells_per_edge);
  fill_random(diff_multfac_smag, num_lev);
  fill_random(kh_c, num_cells*num_lev);
  fill_random(div, num_lev*num_cells);

  setup_mo_nh_diffusion_stencil_03(&gpu_tri_mesh, num_lev, (cudaStream_t) 0);  

  std::vector<double> times;
  for (int i = 0; i < num_runs; i++) {
    double time = run_and_time(run_mo_nh_diffusion_stencil_03, kh_smag_ec, vn, e_bln_c_s,
          geofac_div, diff_multfac_smag, kh_c, div);    
    times.push_back(time);    
  }
  auto avg = [] (const std::vector<double>& in) {return std::accumulate( in.begin(), in.end(), 0.0) / in.size();};
  auto sd = [&avg] (const std::vector<double>& in) {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean);};
  printf("mo_nh_diffusion_stencil_03 num cells %d: %e %e\n", num_cells, avg(times), sd(times));  

  return 0;
}