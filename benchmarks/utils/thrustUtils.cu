#include "thrustUtils.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

struct prg {    
    __host__ __device__
        double operator()(const unsigned int n) const {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<double> dist(0, 1);
            rng.discard(n);            
            return dist(rng);
        }
};

template<typename T>
struct absolute_value {
  __host__ __device__ T operator()(const T &x) const {
    return x < T(0) ? -x : x;
  }
};

void fill_random(double *dev_field, int num_el) {
  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(dev_field);  
  thrust::counting_iterator<unsigned int> index_sequence_begin(0);

  thrust::transform(index_sequence_begin, index_sequence_begin + num_el,
            dev_ptr, prg());
}

bool verify(double *left, double *right, int num_el, double tolerance) {
  thrust::device_vector<double> diff(num_el);
  thrust::device_ptr<double> left_p = thrust::device_pointer_cast(left);  
  thrust::device_ptr<double> right_p = thrust::device_pointer_cast(right);  
  thrust::transform(left_p, left_p + num_el, right_p, diff.begin(), thrust::minus<double>{});
  thrust::transform(diff.begin(), diff.end(), diff.begin(), absolute_value<double>{});
  thrust::device_vector<double>::iterator max_iter = thrust::max_element(diff.begin(), diff.end());
  double max_error = *max_iter;
  printf("MAX ERROR: %e\n", max_error);
  return max_error < tolerance;
}