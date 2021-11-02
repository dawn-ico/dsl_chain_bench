#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"

#include <numeric>

#include "solve_nonhydro_stencil_50.h"

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
  const int num_runs = 1e4;

  const size_t num_edges = gpu_tri_mesh.NumEdges;
  const size_t num_cells = gpu_tri_mesh.NumCells;
  const size_t num_vertices = gpu_tri_mesh.NumVertices; 
  
  double *z_q;
  double *w;

  cudaMalloc((void**) &z_q, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &w, num_cells*num_lev*sizeof(double));  

  fill_random(z_q, num_cells*num_lev);
  fill_random(w, num_cells*num_lev);   

  setup_mo_solve_nonhydro_stencil_50(&gpu_tri_mesh, num_lev, (cudaStream_t) 0);  

  std::vector<double> times;
  for (int i = 0; i < num_runs; i++) {
    double time = run_and_time(run_mo_solve_nonhydro_stencil_50, z_q, w);
    times.push_back(time);    
  }
  auto avg = [] (const std::vector<double>& in) {return std::accumulate( in.begin(), in.end(), 0.0) / in.size();};
  auto sd = [&avg] (const std::vector<double>& in) {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean);};
  printf("mo_solve_nonhydro_stencil_50 num cells %d: %e %e\n", num_cells, avg(times), sd(times));  

  return 0;
}