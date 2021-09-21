#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/to_vtk.h"
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include "driver-includes/math.hpp"
#include <chrono>
#define BLOCK_SIZE 512
#define LEVELS_PER_THREAD 16
using namespace gridtools::dawn;

namespace dawn_generated {
namespace cuda_ico {
__global__ void
k_cache_inline_stencil52_ms76_s77_kernel(int CellStride, int kSize, int hOffset, int hSize,
                                         const ::dawn::float_type* __restrict__ z_w_con_c,
                                         const ::dawn::float_type* __restrict__ w,
                                         const ::dawn::float_type* __restrict__ coeff1_dwdz,
                                         const ::dawn::float_type* __restrict__ coeff2_dwdz,
                                         ::dawn::float_type* __restrict__ ddt_w_adv) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 1;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 1;
  if(pidx >= hSize) {
    return;
  }
  pidx += hOffset;

  ::dawn::float_type w_kcache[3];

  // pre fill caches
  w_kcache[0] = w[(klo + -1) * CellStride + pidx];
  w_kcache[1] = w[(klo + 0) * CellStride + pidx];

  for(int kIter = klo; kIter < khi; kIter++) {
    if(kIter >= kSize + 0) {
      return;
    }

    // head fill cache
    w_kcache[2] = w[(kIter + 1) * CellStride + pidx];

    ddt_w_adv[(kIter + 0) * CellStride + pidx] =
        ((-z_w_con_c[(kIter + 0) * CellStride + pidx]) *
         (((w_kcache[0] * coeff1_dwdz[(kIter + 0) * CellStride + pidx]) -
           (w_kcache[2] * coeff2_dwdz[(kIter + 0) * CellStride + pidx])) +
          (w_kcache[1] * (coeff2_dwdz[(kIter + 0) * CellStride + pidx] -
                                                 coeff1_dwdz[(kIter + 0) * CellStride + pidx]))));

    // slide k chaches
    w_kcache[0] = w_kcache[1];
    w_kcache[1] = w_kcache[2];                                                 
  }
}

class k_cache_inline {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    dawn::unstructured_domain DomainLower;
    dawn::unstructured_domain DomainUpper;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh* mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      DomainLower = mesh->DomainLower;
      DomainUpper = mesh->DomainUpper;
    }
  };

  struct stencil_52 {
  private:
    ::dawn::float_type* z_w_con_c_;
    ::dawn::float_type* w_;
    ::dawn::float_type* coeff1_dwdz_;
    ::dawn::float_type* coeff2_dwdz_;
    ::dawn::float_type* ddt_w_adv_;
    static int kSize_;
    static GpuTriMesh mesh_;
    static bool is_setup_;
    static cudaStream_t stream_;

  public:
    static const GpuTriMesh& getMesh() { return mesh_; }

    static int getKSize() { return kSize_; }

    static void free() {}

    static void setup(const dawn::GlobalGpuTriMesh* mesh, int kSize, cudaStream_t stream) {
      mesh_ = GpuTriMesh(mesh);
      kSize_ = kSize;
      is_setup_ = true;
      stream_ = stream;
    }

    dim3 grid(int kSize, int elSize, bool kparallel) {
      if(kparallel) {
        int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
        return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, dK, 1);
      } else {
        return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
      }
    }

    stencil_52() {}

    void run() {
      if(!is_setup_) {
        printf("k_cache_inline has not been set up! make sure setup() is called before run!\n");
        return;
      }
      dim3 dB(BLOCK_SIZE, 1, 1);
      int hsize77 =
          mesh_.DomainUpper({::dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Halo, 0}) -
          mesh_.DomainLower(
              {::dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0}) +
          1;
      if(hsize77 == 0) {
        return;
      }
      int hoffset77 =
          mesh_.DomainLower({::dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0});
      dim3 dG77 = grid(kSize_ + 0 - 1, hsize77, true);
      k_cache_inline_stencil52_ms76_s77_kernel<<<dG77, dB, 0, stream_>>>(
          mesh_.CellStride, kSize_, hoffset77, hsize77, z_w_con_c_, w_, coeff1_dwdz_, coeff2_dwdz_,
          ddt_w_adv_);
#ifndef NDEBUG

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    void CopyResultToHost(::dawn::float_type* ddt_w_adv, bool do_reshape) {
      if(do_reshape) {
        ::dawn::float_type* host_buf = new ::dawn::float_type[(mesh_.CellStride) * kSize_];
        gpuErrchk(cudaMemcpy((::dawn::float_type*)host_buf, ddt_w_adv_,
                             (mesh_.CellStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, ddt_w_adv, kSize_, mesh_.CellStride);
        delete[] host_buf;
      } else {
        gpuErrchk(cudaMemcpy(ddt_w_adv, ddt_w_adv_,
                             (mesh_.CellStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
    }

    void copy_memory(::dawn::float_type* z_w_con_c, ::dawn::float_type* w,
                     ::dawn::float_type* coeff1_dwdz, ::dawn::float_type* coeff2_dwdz,
                     ::dawn::float_type* ddt_w_adv, bool do_reshape) {
      dawn::initField(z_w_con_c, &z_w_con_c_, mesh_.CellStride, kSize_, do_reshape);
      dawn::initField(w, &w_, mesh_.CellStride, kSize_, do_reshape);
      dawn::initField(coeff1_dwdz, &coeff1_dwdz_, mesh_.CellStride, kSize_, do_reshape);
      dawn::initField(coeff2_dwdz, &coeff2_dwdz_, mesh_.CellStride, kSize_, do_reshape);
      dawn::initField(ddt_w_adv, &ddt_w_adv_, mesh_.CellStride, kSize_, do_reshape);
    }

    void copy_pointers(::dawn::float_type* z_w_con_c, ::dawn::float_type* w,
                       ::dawn::float_type* coeff1_dwdz, ::dawn::float_type* coeff2_dwdz,
                       ::dawn::float_type* ddt_w_adv) {
      z_w_con_c_ = z_w_con_c;
      w_ = w;
      coeff1_dwdz_ = coeff1_dwdz;
      coeff2_dwdz_ = coeff2_dwdz;
      ddt_w_adv_ = ddt_w_adv;
    }
  };
};
} // namespace cuda_ico
} // namespace dawn_generated
extern "C" {
void run_k_cache_inline_from_c_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                    ::dawn::float_type* z_w_con_c, ::dawn::float_type* w,
                                    ::dawn::float_type* coeff1_dwdz,
                                    ::dawn::float_type* coeff2_dwdz,
                                    ::dawn::float_type* ddt_w_adv) {
  dawn_generated::cuda_ico::k_cache_inline::stencil_52 s;
  dawn_generated::cuda_ico::k_cache_inline::stencil_52::setup(mesh, k_size, 0);
  s.copy_memory(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv, true);
  s.run();
  s.CopyResultToHost(ddt_w_adv, true);
  dawn_generated::cuda_ico::k_cache_inline::stencil_52::free();
  return;
}
void run_k_cache_inline_from_fort_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                       ::dawn::float_type* z_w_con_c, ::dawn::float_type* w,
                                       ::dawn::float_type* coeff1_dwdz,
                                       ::dawn::float_type* coeff2_dwdz,
                                       ::dawn::float_type* ddt_w_adv) {
  dawn_generated::cuda_ico::k_cache_inline::stencil_52 s;
  dawn_generated::cuda_ico::k_cache_inline::stencil_52::setup(mesh, k_size, 0);
  s.copy_memory(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv, false);
  s.run();
  s.CopyResultToHost(ddt_w_adv, false);
  dawn_generated::cuda_ico::k_cache_inline::stencil_52::free();
  return;
}
void run_k_cache_inline(::dawn::float_type* z_w_con_c, ::dawn::float_type* w,
                        ::dawn::float_type* coeff1_dwdz, ::dawn::float_type* coeff2_dwdz,
                        ::dawn::float_type* ddt_w_adv) {
  dawn_generated::cuda_ico::k_cache_inline::stencil_52 s;
  s.copy_pointers(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv);
  s.run();
  return;
}
bool verify_k_cache_inline(const ::dawn::float_type* ddt_w_adv_dsl,
                           const ::dawn::float_type* ddt_w_adv, const double ddt_w_adv_rel_tol,
                           const double ddt_w_adv_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto& mesh = dawn_generated::cuda_ico::k_cache_inline::stencil_52::getMesh();
  int kSize = dawn_generated::cuda_ico::k_cache_inline::stencil_52::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  bool isValid;
  isValid = ::dawn::verify_field((mesh.CellStride) * kSize, ddt_w_adv_dsl, ddt_w_adv, "ddt_w_adv",
                                 ddt_w_adv_rel_tol, ddt_w_adv_abs_tol);
  if(!isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), kSize, (mesh.CellStride), ddt_w_adv,
                          "k_cache_inline", "ddt_w_adv", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), kSize, (mesh.CellStride), ddt_w_adv_dsl,
                          "k_cache_inline", "ddt_w_adv_dsl", iteration);
    std::cout << "[DSL] serializing ddt_w_adv as error is high.\n" << std::flush;
#endif
  }
#ifdef __SERIALIZE_ON_ERROR

      serialize_flush_iter("k_cache_inline", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n" << std::flush;
  return isValid;
}
void run_and_verify_k_cache_inline(::dawn::float_type* z_w_con_c, ::dawn::float_type* w,
                                   ::dawn::float_type* coeff1_dwdz, ::dawn::float_type* coeff2_dwdz,
                                   ::dawn::float_type* ddt_w_adv,
                                   ::dawn::float_type* ddt_w_adv_before,
                                   const double ddt_w_adv_rel_tol, const double ddt_w_adv_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil k_cache_inline (" << iteration << ") ...\n" << std::flush;
  run_k_cache_inline(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv_before);
  std::cout << "[DSL] k_cache_inline run time: " << time << "s\n" << std::flush;
  std::cout << "[DSL] Verifying stencil k_cache_inline...\n" << std::flush;
  verify_k_cache_inline(ddt_w_adv_before, ddt_w_adv, ddt_w_adv_rel_tol, ddt_w_adv_abs_tol,
                        iteration);
  iteration++;
}
void setup_k_cache_inline(dawn::GlobalGpuTriMesh* mesh, int k_size, cudaStream_t stream) {
  dawn_generated::cuda_ico::k_cache_inline::stencil_52::setup(mesh, k_size, stream);
}
void free_k_cache_inline() { dawn_generated::cuda_ico::k_cache_inline::stencil_52::free(); }
}
int dawn_generated::cuda_ico::k_cache_inline::stencil_52::kSize_;
cudaStream_t dawn_generated::cuda_ico::k_cache_inline::stencil_52::stream_;
bool dawn_generated::cuda_ico::k_cache_inline::stencil_52::is_setup_ = false;
dawn_generated::cuda_ico::k_cache_inline::GpuTriMesh
    dawn_generated::cuda_ico::k_cache_inline::stencil_52::mesh_;

