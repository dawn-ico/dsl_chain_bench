#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"

#include <numeric>

#include "nh_diffusion_stencil_01.h"

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
   
  // atlas::Mesh mesh = *meshptr;
  dawn::GlobalGpuTriMesh gpu_tri_mesh = atlasToGlobalGpuTriMesh(mesh);
  const int num_lev = 65;
  const int num_runs = 100000;

  const size_t num_edges = gpu_tri_mesh.NumEdges;
  const size_t num_cells = gpu_tri_mesh.NumCells;
  const size_t num_vertices = gpu_tri_mesh.NumVertices;
  const size_t diamond_size = 4;
  
  double *diff_multfac_smag;
  double *tangent_orientation;
  double *inv_primal_edge_length;
  double *inv_vert_vert_length;
  double *u_vert;
  double *v_vert;
  double *primal_normal_vert_x;
  double *primal_normal_vert_y;
  double *dual_normal_vert_x;
  double *dual_normal_vert_y;
  double *vn;
  double *smag_limit;
  double *kh_smag_e;
  double *kh_smag_ec;
  double *z_nabla2_e;

  cudaMalloc((void**) &diff_multfac_smag, num_lev*sizeof(double));
  cudaMalloc((void**) &tangent_orientation, num_edges*sizeof(double));
  cudaMalloc((void**) &inv_primal_edge_length, num_edges*sizeof(double));
  cudaMalloc((void**) &inv_vert_vert_length, num_edges*sizeof(double));
  cudaMalloc((void**) &u_vert, num_vertices*num_lev*sizeof(double));
  cudaMalloc((void**) &v_vert, num_vertices*num_lev*sizeof(double));
  cudaMalloc((void**) &primal_normal_vert_x, num_edges*diamond_size*sizeof(double));
  cudaMalloc((void**) &primal_normal_vert_y, num_edges*diamond_size*sizeof(double));
  cudaMalloc((void**) &dual_normal_vert_x, num_edges*diamond_size*sizeof(double));
  cudaMalloc((void**) &dual_normal_vert_y, num_edges*diamond_size*sizeof(double));
  cudaMalloc((void**) &vn, num_edges*num_lev*sizeof(double));
  cudaMalloc((void**) &smag_limit, num_lev*sizeof(double));
  cudaMalloc((void**) &kh_smag_e, num_edges*num_lev*sizeof(double));
  cudaMalloc((void**) &kh_smag_ec, num_edges*num_lev*sizeof(double));
  cudaMalloc((void**) &z_nabla2_e, num_edges*num_lev*sizeof(double));

  fill_random(diff_multfac_smag, num_lev);
  fill_random(tangent_orientation, num_edges);
  fill_random(inv_primal_edge_length, num_edges);
  fill_random(inv_vert_vert_length, num_edges);
  fill_random(u_vert, num_vertices*num_lev);
  fill_random(v_vert, num_vertices*num_lev);
  fill_random(primal_normal_vert_x, num_edges*diamond_size);
  fill_random(primal_normal_vert_y, num_edges*diamond_size);
  fill_random(dual_normal_vert_x, num_edges*diamond_size);
  fill_random(dual_normal_vert_y, num_edges*diamond_size);
  fill_random(vn, num_edges);
  fill_random(smag_limit, num_lev);
  fill_random(kh_smag_e, num_edges*num_lev);
  fill_random(kh_smag_ec, num_edges*num_lev);
  fill_random(z_nabla2_e, num_edges*num_lev);  

  setup_mo_nh_diffusion_stencil_01(&gpu_tri_mesh, num_lev, (cudaStream_t) 0);  

  double smag_offset = 42.;

  std::vector<double> times;
  for (int i = 0; i < num_runs; i++) {
    double time = run_and_time(run_mo_nh_diffusion_stencil_01, smag_offset, 
          diff_multfac_smag, tangent_orientation, inv_primal_edge_length,
          inv_vert_vert_length, u_vert, v_vert, primal_normal_vert_x, primal_normal_vert_y,
          dual_normal_vert_x,dual_normal_vert_y, vn, smag_limit,
          kh_smag_e, kh_smag_ec, z_nabla2_e);    
    times.push_back(time);    
  }
  auto avg = [] (const std::vector<double>& in) {return std::accumulate( in.begin(), in.end(), 0.0) / in.size();};
  auto sd = [&avg] (const std::vector<double>& in) {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean);};
  printf("mo_nh_diffusion_stencil_01 num cells %d: %e %e\n", num_cells, avg(times), sd(times));  

  return 0;
}