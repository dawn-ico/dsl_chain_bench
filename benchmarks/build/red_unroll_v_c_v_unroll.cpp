#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/to_vtk.h"
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include "driver-includes/math.hpp"
#include <chrono>
#define BLOCK_SIZE 128
#define LEVELS_PER_THREAD 1
using namespace gridtools::dawn;

namespace dawn_generated {
namespace cuda_ico {
template <int C_V_SIZE, int V_C_SIZE>
__global__ void unroll_v_c_v_unroll_stencil37_ms46_s47_kernel(
    int VertexStride, int CellStride, int kSize, int hOffset, int hSize, const int* cvTable,
    const int* vcTable, const ::dawn::float_type* __restrict__ kh_smag_e,
    const ::dawn::float_type* __restrict__ inv_dual_edge_length,
    const ::dawn::float_type* __restrict__ theta_v, ::dawn::float_type* __restrict__ outF) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if(pidx >= hSize) {
    return;
  }
  pidx += hOffset;
  for(int kIter = klo; kIter < khi; kIter++) {
    if(kIter >= kSize + 0) {
      return;
    }
    ::dawn::float_type lhs_62 = (::dawn::float_type)0;
    for(int nbhIter0 = 0; nbhIter0 < V_C_SIZE; nbhIter0++) {
      int nbhIdx0 = vcTable[pidx + VertexStride * nbhIter0];
      if(nbhIdx0 == DEVICE_MISSING_VALUE) {
        continue;
      }
      ::dawn::float_type lhs_57 = (::dawn::float_type)0;
      for(int nbhIter1 = 0; nbhIter1 < C_V_SIZE; nbhIter1++) {
        int nbhIdx1 = cvTable[nbhIdx0 + CellStride * nbhIter1];
        lhs_57 += theta_v[(kIter + 0) * VertexStride + nbhIdx1];
      }
      lhs_62 += ((kh_smag_e[(kIter + 0) * CellStride + nbhIdx0] * inv_dual_edge_length[nbhIdx0]) *
                 lhs_57);
    }
    outF[(kIter + 0) * VertexStride + pidx] = lhs_62;
  }
}

class unroll_v_c_v_unroll {
public:
  static const int V_C_SIZE = 6;
  static const int C_V_SIZE = 3;

  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    dawn::unstructured_domain DomainLower;
    dawn::unstructured_domain DomainUpper;
    int* vcTable;
    int* cvTable;

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
      vcTable = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{
          {dawn::LocationType::Vertices, dawn::LocationType::Cells}, 0});
      cvTable = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{
          {dawn::LocationType::Cells, dawn::LocationType::Vertices}, 0});
    }
  };

  struct stencil_37 {
  private:
    ::dawn::float_type* kh_smag_e_;
    ::dawn::float_type* inv_dual_edge_length_;
    ::dawn::float_type* theta_v_;
    ::dawn::float_type* outF_;
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

    stencil_37() {}

    void run() {
      if(!is_setup_) {
        printf(
            "unroll_v_c_v_unroll has not been set up! make sure setup() is called before run!\n");
        return;
      }
      dim3 dB(BLOCK_SIZE, 1, 1);
      int hsize47 = mesh_.DomainUpper(
                        {::dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Halo, 0}) -
                    mesh_.DomainLower(
                        {::dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Nudging, 0}) +
                    1;
      if(hsize47 == 0) {
        return;
      }
      int hoffset47 = mesh_.DomainLower(
          {::dawn::LocationType::Vertices, dawn::UnstructuredSubdomain::Nudging, 0});
      dim3 dG47 = grid(kSize_ + 0 - 0, hsize47, true);
      unroll_v_c_v_unroll_stencil37_ms46_s47_kernel<C_V_SIZE, V_C_SIZE><<<dG47, dB, 0, stream_>>>(
          mesh_.VertexStride, mesh_.CellStride, kSize_, hoffset47, hsize47, mesh_.cvTable,
          mesh_.vcTable, kh_smag_e_, inv_dual_edge_length_, theta_v_, outF_);
#ifndef NDEBUG

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    void CopyResultToHost(::dawn::float_type* outF, bool do_reshape) {
      if(do_reshape) {
        ::dawn::float_type* host_buf = new ::dawn::float_type[(mesh_.VertexStride) * kSize_];
        gpuErrchk(cudaMemcpy((::dawn::float_type*)host_buf, outF_,
                             (mesh_.VertexStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, outF, kSize_, mesh_.VertexStride);
        delete[] host_buf;
      } else {
        gpuErrchk(cudaMemcpy(outF, outF_,
                             (mesh_.VertexStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
    }

    void copy_memory(::dawn::float_type* kh_smag_e, ::dawn::float_type* inv_dual_edge_length,
                     ::dawn::float_type* theta_v, ::dawn::float_type* outF, bool do_reshape) {
      dawn::initField(kh_smag_e, &kh_smag_e_, mesh_.CellStride, kSize_, do_reshape);
      dawn::initField(inv_dual_edge_length, &inv_dual_edge_length_, mesh_.CellStride, 1,
                      do_reshape);
      dawn::initField(theta_v, &theta_v_, mesh_.VertexStride, kSize_, do_reshape);
      dawn::initField(outF, &outF_, mesh_.VertexStride, kSize_, do_reshape);
    }

    void copy_pointers(::dawn::float_type* kh_smag_e, ::dawn::float_type* inv_dual_edge_length,
                       ::dawn::float_type* theta_v, ::dawn::float_type* outF) {
      kh_smag_e_ = kh_smag_e;
      inv_dual_edge_length_ = inv_dual_edge_length;
      theta_v_ = theta_v;
      outF_ = outF;
    }
  };
};
} // namespace cuda_ico
} // namespace dawn_generated
extern "C" {
void run_unroll_v_c_v_unroll_from_c_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                         ::dawn::float_type* kh_smag_e,
                                         ::dawn::float_type* inv_dual_edge_length,
                                         ::dawn::float_type* theta_v, ::dawn::float_type* outF) {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37 s;
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::setup(mesh, k_size, 0);
  s.copy_memory(kh_smag_e, inv_dual_edge_length, theta_v, outF, true);
  s.run();
  s.CopyResultToHost(outF, true);
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::free();
  return;
}
void run_unroll_v_c_v_unroll_from_fort_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                            ::dawn::float_type* kh_smag_e,
                                            ::dawn::float_type* inv_dual_edge_length,
                                            ::dawn::float_type* theta_v, ::dawn::float_type* outF) {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37 s;
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::setup(mesh, k_size, 0);
  s.copy_memory(kh_smag_e, inv_dual_edge_length, theta_v, outF, false);
  s.run();
  s.CopyResultToHost(outF, false);
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::free();
  return;
}
void run_unroll_v_c_v_unroll(::dawn::float_type* kh_smag_e,
                             ::dawn::float_type* inv_dual_edge_length, ::dawn::float_type* theta_v,
                             ::dawn::float_type* outF) {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37 s;
  s.copy_pointers(kh_smag_e, inv_dual_edge_length, theta_v, outF);
  s.run();
  return;
}
bool verify_unroll_v_c_v_unroll(const ::dawn::float_type* outF_dsl, const ::dawn::float_type* outF,
                                const double outF_rel_tol, const double outF_abs_tol,
                                const int iteration) {
  using namespace std::chrono;
  const auto& mesh = dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::getMesh();
  int kSize = dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  bool isValid;
  isValid = ::dawn::verify_field((mesh.VertexStride) * kSize, outF_dsl, outF, "outF", outF_rel_tol,
                                 outF_abs_tol);
  if(!isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_verts(0, (mesh.NumVertices - 1), kSize, (mesh.VertexStride), outF,
                          "unroll_v_c_v_unroll", "outF", iteration);
    serialize_dense_verts(0, (mesh.NumVertices - 1), kSize, (mesh.VertexStride), outF_dsl,
                          "unroll_v_c_v_unroll", "outF_dsl", iteration);
    std::cout << "[DSL] serializing outF as error is high.\n" << std::flush;
#endif
  }
#ifdef __SERIALIZE_ON_ERROR

      serialize_flush_iter("unroll_v_c_v_unroll", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n" << std::flush;
  return isValid;
}
void run_and_verify_unroll_v_c_v_unroll(::dawn::float_type* kh_smag_e,
                                        ::dawn::float_type* inv_dual_edge_length,
                                        ::dawn::float_type* theta_v, ::dawn::float_type* outF,
                                        ::dawn::float_type* outF_before, const double outF_rel_tol,
                                        const double outF_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil unroll_v_c_v_unroll (" << iteration << ") ...\n"
            << std::flush;
  run_unroll_v_c_v_unroll(kh_smag_e, inv_dual_edge_length, theta_v, outF_before);
  std::cout << "[DSL] unroll_v_c_v_unroll run time: " << time << "s\n" << std::flush;
  std::cout << "[DSL] Verifying stencil unroll_v_c_v_unroll...\n" << std::flush;
  verify_unroll_v_c_v_unroll(outF_before, outF, outF_rel_tol, outF_abs_tol, iteration);
  iteration++;
}
void setup_unroll_v_c_v_unroll(dawn::GlobalGpuTriMesh* mesh, int k_size, cudaStream_t stream) {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::setup(mesh, k_size, stream);
}
void free_unroll_v_c_v_unroll() {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::free();
}
}
int dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::kSize_;
cudaStream_t dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::stream_;
bool dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::is_setup_ = false;
dawn_generated::cuda_ico::unroll_v_c_v_unroll::GpuTriMesh
    dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::mesh_;

