#include "atlasToGlobalGpuTriMesh.h"
#include "thrustUtils.h"

#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"

#include <numeric>

#include "solve_nonhydro_stencil_10.h"

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
  
  double *w;
  double *w_concorr_c;
  double *ddqz_z_half;
  double *rho_now;
  double *rho_var;
  double *theta_now;
  double *theta_var;
  double *wgtfac_c;
  double *theta_ref_mc;
  double *vwind_expl_wgt;
  double *exner_pr;
  double *d_exner_dz_ref_ic;
  double *rho_ic;
  double *z_theta_v_pr_ic;  
  double *theta_v_ic;
  double *z_th_ddz_exner_c;

  cudaMalloc((void**) &w, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &w_concorr_c, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &ddqz_z_half, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &rho_now, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &rho_var, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &theta_now, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &theta_var, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &wgtfac_c, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &theta_ref_mc, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &vwind_expl_wgt, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &exner_pr, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &d_exner_dz_ref_ic, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &rho_ic, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &z_theta_v_pr_ic, num_cells*num_lev*sizeof(double));  
  cudaMalloc((void**) &theta_v_ic, num_cells*num_lev*sizeof(double));
  cudaMalloc((void**) &z_th_ddz_exner_c, num_cells*num_lev*sizeof(double));  

  fill_random(w, num_cells*num_lev);
  fill_random(w_concorr_c, num_cells*num_lev);
  fill_random(ddqz_z_half, num_cells*num_lev);
  fill_random(rho_now, num_cells*num_lev);
  fill_random(rho_var, num_cells*num_lev);
  fill_random(theta_now, num_cells*num_lev);
  fill_random(theta_var, num_cells*num_lev);
  fill_random(wgtfac_c, num_cells*num_lev);
  fill_random(theta_ref_mc, num_cells*num_lev);
  fill_random(vwind_expl_wgt, num_cells*num_lev);
  fill_random(exner_pr, num_cells*num_lev);
  fill_random(d_exner_dz_ref_ic, num_cells*num_lev);
  fill_random(rho_ic, num_cells*num_lev);
  fill_random(z_theta_v_pr_ic, num_cells*num_lev);  
  fill_random(theta_v_ic, num_cells*num_lev);
  fill_random(z_th_ddz_exner_c, num_cells*num_lev);  


  double dtime = 42.;
  double wgt_nnow_rth = 42.;
  double wgt_nnew_rth = 42.;

  setup_mo_solve_nonhydro_stencil_10(&gpu_tri_mesh, num_lev, (cudaStream_t) 0);  

  std::vector<double> times;
  for (int i = 0; i < num_runs; i++) {
    double time = run_and_time(run_mo_solve_nonhydro_stencil_10, dtime, wgt_nnow_rth, wgt_nnew_rth, 
                    w,  w_concorr_c, ddqz_z_half, rho_now, rho_var, theta_now, theta_var, wgtfac_c,
                    theta_ref_mc, vwind_expl_wgt, exner_pr, d_exner_dz_ref_ic, rho_ic, z_theta_v_pr_ic,  
                    theta_v_ic, z_th_ddz_exner_c);
    times.push_back(time);    
  }
  auto avg = [] (const std::vector<double>& in) {return std::accumulate( in.begin(), in.end(), 0.0) / in.size();};
  auto sd = [&avg] (const std::vector<double>& in) {
                  double mean = avg(in);
                  double sq_sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
                  return std::sqrt(sq_sum / in.size() - mean * mean);};
  printf("mo_solve_nonhydro_stencil_10 num cells %d: %e %e\n", num_cells, avg(times), sd(times));  

  return 0;
}