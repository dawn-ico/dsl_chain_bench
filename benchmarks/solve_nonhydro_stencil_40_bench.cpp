#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"

#include <numeric>

#include "solve_nonhydro_stencil_40.h"

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
  
  double *z_w_expl;
  double *w_nnow;
  double *ddt_w_adv_ntl1;
  double *ddt_w_adv_ntl2;
  double *z_th_ddz_exner_c;
  double *z_contr_w_fl_l;
  double *rho_ic;
  double *w_concorr_c;
  double *vwind_expl_wgt;

  cudaMalloc((void**) &z_w_expl, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &w_nnow, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &ddt_w_adv_ntl1, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &ddt_w_adv_ntl2, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &z_th_ddz_exner_c, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &z_contr_w_fl_l, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &rho_ic, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &w_concorr_c, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &vwind_expl_wgt, num_cells*num_lev*sizeof(double));

  fill_random(z_w_expl, num_cells*num_lev);
  fill_random(w_nnow, num_cells*num_lev);
  fill_random(ddt_w_adv_ntl1, num_cells*num_lev);
  fill_random(ddt_w_adv_ntl2, num_cells*num_lev);
  fill_random(z_th_ddz_exner_c, num_cells*num_lev);
  fill_random(z_contr_w_fl_l, num_cells*num_lev);
  fill_random(rho_ic, num_cells*num_lev);
  fill_random(w_concorr_c, num_cells*num_lev);
  fill_random(vwind_expl_wgt, num_cells*num_lev);

  double dtime = 42.;
  double wgt_nnow_rth = 42.;
  double wgt_nnew_rth = 42.;
  double cpd = 42.;

  setup_mo_solve_nonhydro_stencil_40(&gpu_tri_mesh, num_lev, (cudaStream_t) 0);  

  std::vector<double> times;
  for (int i = 0; i < num_runs; i++) {
    double time = run_and_time(run_mo_solve_nonhydro_stencil_40, dtime, wgt_nnow_rth, wgt_nnew_rth, cpd,
                    z_w_expl,  w_nnow, ddt_w_adv_ntl1, ddt_w_adv_ntl2, z_th_ddz_exner_c, z_contr_w_fl_l, rho_ic, w_concorr_c,
                    vwind_expl_wgt);
    times.push_back(time);    
  }
  auto avg = [] (const std::vector<double>& in) {return std::accumulate( in.begin(), in.end(), 0.0) / in.size();};
  auto sd = [&avg] (const std::vector<double>& in) {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean);};
  printf("mo_solve_nonhydro_stencil_40 num cells %d: %e %e\n", num_cells, avg(times), sd(times));  

  return 0;
}