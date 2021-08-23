#include "atlasToGlobalGpuTriMesh.h"

struct prg {    
    __host__ __device__
        double operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<double> dist(0, 1);
            rng.discard(n);
            return dist(rng);
        }
};

void fill_randon(double *dev_field, int num_el) {
  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(dev_field);  
  thrust::counting_iterator<unsigned int> index_sequence_begin(0);

  thrust::transform(index_sequence_begin, index_sequence_begin + num_el,
            dev_ptr, prg());
}

bool verify(double *left, double *right, int num_el, double tolerance) {
  thrust::device_vector<double> diff(num_el);
  thrust::devive_ptr<double> left_p = thrust::device_pointer_cast(left);  
  thrust::devive_ptr<double> right_p = thrust::device_pointer_cast(right);  
  thrust::transform(left_p, left_p + num_el, right_p, diff.begin(), thrust::minus<double>)
  thrust::transform(diff.begin(), diff.end(), diff.begin(), thrust::abs<double>);
  thrust::device_vector<double>::iterator max_iter = thrust::max_element(diff.begin(), diff.end());
  return *max_iter < tolerance;
}

template<typename... Args>
double run_and_time(void (*fun) (Args... args), Args... args) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  fun(args...);
  cudaEventSynchronize(stop);
  double milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  return milliseconds;
}

int main() {
  atlas::Mesh mesh = AtlasMeshFromNetCDFComplete("grid.nc");
  dawn::GlobalGpuTriMesh gpu_tri_mesh = atlasToGlobalGpuTriMesh(mesh);
  const int num_lev = 80;

  const size_t in_size = sizeof(double)*mesh.nodes().size()*num_lev;
  const size_t out_size = sizeof(double)*mesh.edges().size()*num_lev;
  double *in_field_nested, *out_field_nested;
  double *in_field_sequential, *out_field_sequential;
  
  cudaMalloc((void**)&in_field_nested, in_size);
  cudaMalloc((void**)&out_field_nested, out_size);
  cudaMalloc((void**)&in_field_sequential, in_size);
  cudaMalloc((void**)&out_field_sequential, out_size);

  fill_random(in_field_nested, in_size);
  cudaMemcpy(in_field_sequential, in_field_nested, in_size, cudaMemcpyDeviceToDevice);

  cudaStream_t stream;
  setup_red_e_c_v_inlined(gpu_tri_mesh, num_lev, stream);
  setup_red_e_c_v_sequential(gpu_tri_mesh, num_lev, stream);

  double time_nested = run_and_time(run_e_c_v_nested, in_field_nested, out_field_nested);
  double time_sequential = run_and_time(run_e_c_v_sequential, in_field_sequential, out_field_sequential);

  printf("E > C > V: seq %e nest %e\n", time_nested, time_sequential);

  bool verify(out_field_nested, out_field_sequential);
  if (!verify) {
    printf("[FAIL] Failed Verification!")
  }

  return 0;
}