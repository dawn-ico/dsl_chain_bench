#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/to_vtk.h"
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include "driver-includes/math.hpp"
#include <chrono>
#include <set>
#include <assert.h>
#define BLOCK_SIZE 128
#define LEVELS_PER_THREAD 1
using namespace gridtools::dawn;

namespace dawn_generated {
namespace cuda_ico {
__global__ void
vcv_kernel(int VertexStride, int CellStride, int kSize, int hOffset, int hSize,
           const int *vcTable, const int *vvTable,
           const ::dawn::float_type *__restrict__ inv_dual_edge_length,
           const ::dawn::float_type *__restrict__ kh_smag_e,
           const ::dawn::float_type *__restrict__ theta_v,
           ::dawn::float_type *__restrict__ z_temp) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD;
  int khi = (kidx + 1) * LEVELS_PER_THREAD;
  if (pidx >= hSize) {
    return;
  }
  pidx += hOffset;
  for (int kIter = klo; kIter < khi; kIter++) {
    if (kIter >= kSize) {
      return;
    }

    const int nbhIdx0_0 = vcTable[pidx + VertexStride * 0];
    const int nbhIdx0_1 = vcTable[pidx + VertexStride * 1];
    const int nbhIdx0_2 = vcTable[pidx + VertexStride * 2];
    const int nbhIdx0_3 = vcTable[pidx + VertexStride * 3];
    const int nbhIdx0_4 = vcTable[pidx + VertexStride * 4];
    const int nbhIdx0_5 = vcTable[pidx + VertexStride * 5];

    const int nbhIdx1_0 = vvTable[pidx + VertexStride * 0];
    const int nbhIdx1_1 = vvTable[pidx + VertexStride * 1];
    const int nbhIdx1_2 = vvTable[pidx + VertexStride * 2];
    const int nbhIdx1_3 = vvTable[pidx + VertexStride * 3];
    const int nbhIdx1_4 = vvTable[pidx + VertexStride * 4];
    const int nbhIdx1_5 = vvTable[pidx + VertexStride * 5];

    int self_idx = kIter * VertexStride + pidx;

    ::dawn::float_type lhs_566 =
        ((kh_smag_e[kIter * CellStride + nbhIdx0_0] *
          inv_dual_edge_length[nbhIdx0_0]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_0] + theta_v[nbhIdx1_1])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_1] *
          inv_dual_edge_length[nbhIdx0_1]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_1] + theta_v[nbhIdx1_2])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_2] *
          inv_dual_edge_length[nbhIdx0_2]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_2] + theta_v[nbhIdx1_3])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_3] *
          inv_dual_edge_length[nbhIdx0_3]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_3] + theta_v[nbhIdx1_4])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_4] *
          inv_dual_edge_length[nbhIdx0_4]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_4] + theta_v[nbhIdx1_5])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_5] *
          inv_dual_edge_length[nbhIdx0_5]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_5] + theta_v[nbhIdx1_0]));

    z_temp[self_idx] = lhs_566;
  }
}

class unroll_v_c_v_unroll {
public:
  static const int V_C_SIZE = 6;
  static const int C_V_SIZE = 3;
  static const int V_V_SIZE = 6;

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
    int* vvTable;

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
    ::dawn::float_type* z_temp_;
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

  int *vvTable_h = new int[V_V_SIZE * mesh_.VertexStride];
  int *vcTable_h = new int[V_C_SIZE * mesh_.VertexStride];
  int *cvTable_h = new int[C_V_SIZE * mesh_.CellStride];

  cudaMemcpy(vcTable_h, mesh_.vcTable,
             sizeof(int) * V_C_SIZE * mesh_.VertexStride, cudaMemcpyDeviceToHost);
  cudaMemcpy(cvTable_h, mesh_.cvTable,
             sizeof(int) * C_V_SIZE * mesh_.CellStride,
             cudaMemcpyDeviceToHost);

  std::fill(vvTable_h, vvTable_h + mesh_.VertexStride * V_V_SIZE, -1);

  for (int elemIdx = 0; elemIdx < mesh_.VertexStride; elemIdx++) {    
    std::vector<std::set<int>> nbhs;
    for (int nbhIter0 = 0; nbhIter0 < V_C_SIZE; nbhIter0++) {
      std::set<int> nbh;
      int nbhIdx0 = vcTable_h[elemIdx + mesh_.VertexStride * nbhIter0];
      for (int nbhIter1 = 0; nbhIter1 < C_V_SIZE; nbhIter1++) {
        int nbhIdx1 = cvTable_h[nbhIdx0 + mesh_.CellStride * nbhIter1];
        if (nbhIdx1 != nbhIdx0) {
          nbh.insert(nbhIdx1);
        }
      }
      nbhs.push_back(nbh);
    }

    std::vector<int> tour;

    int behind = nbhs.size() - 1;
    int cur = 0;
    int ahead = 1;
    while (true) {
      std::set<int> intersect_right;
      set_intersection(nbhs[cur].begin(), nbhs[cur].end(), nbhs[ahead].begin(),
                       nbhs[ahead].end(),
                       std::inserter(intersect_right, intersect_right.begin()));

      std::set<int> intersect_left;
      set_intersection(nbhs[cur].begin(), nbhs[cur].end(), nbhs[behind].begin(),
                       nbhs[behind].end(),
                       std::inserter(intersect_left, intersect_left.begin()));

      assert(intersect_right.size() == 1);
      assert(intersect_left.size() == 1);

      int anchor_left = *intersect_left.begin();
      int anchor_right = *intersect_right.begin();
      for (auto it = nbhs[cur].begin(); it != nbhs[cur].end(); ++it) {
        if (*it != anchor_left && *it != anchor_right) {
          tour.push_back(*it);
        }
      }
      tour.push_back(anchor_right);

      if (ahead == 0) {
        break;
      }

      behind++;
      if (behind == nbhs.size()) {
        behind = 0;
      }
      cur++;
      ahead++;
      if (cur == nbhs.size() - 1) {
        ahead = 0;
      }
    }

    assert(tour.size() == V_V_SIZE);
    for (int lin_idx = 0; lin_idx < V_V_SIZE; lin_idx++) {
      vvTable_h[elemIdx + mesh_.VertexStride * lin_idx] = tour[lin_idx];
    }
   }

  cudaMalloc((void **)&mesh_.vvTable,
          sizeof(int) * mesh_.VertexStride * V_V_SIZE);
  cudaMemcpy(mesh_.vvTable, vvTable_h,
             sizeof(int) * mesh_.VertexStride * V_V_SIZE,
             cudaMemcpyHostToDevice);
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
      vcv_kernel<<<dG47, dB, 0, stream_>>>(
          mesh_.VertexStride, mesh_.CellStride, kSize_, hoffset47, hsize47, mesh_.vcTable,
          mesh_.vvTable, kh_smag_e_, inv_dual_edge_length_, theta_v_, z_temp_);
#ifndef NDEBUG

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    void CopyResultToHost(::dawn::float_type* z_temp, bool do_reshape) {
      if(do_reshape) {
        ::dawn::float_type* host_buf = new ::dawn::float_type[(mesh_.VertexStride) * kSize_];
        gpuErrchk(cudaMemcpy((::dawn::float_type*)host_buf, z_temp_,
                             (mesh_.VertexStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, z_temp, kSize_, mesh_.VertexStride);
        delete[] host_buf;
      } else {
        gpuErrchk(cudaMemcpy(z_temp, z_temp_,
                             (mesh_.VertexStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
    }

    void copy_memory(::dawn::float_type* kh_smag_e, ::dawn::float_type* inv_dual_edge_length,
                     ::dawn::float_type* theta_v, ::dawn::float_type* z_temp, bool do_reshape) {
      dawn::initField(kh_smag_e, &kh_smag_e_, mesh_.CellStride, kSize_, do_reshape);
      dawn::initField(inv_dual_edge_length, &inv_dual_edge_length_, mesh_.CellStride, 1,
                      do_reshape);
      dawn::initField(theta_v, &theta_v_, mesh_.VertexStride, kSize_, do_reshape);
      dawn::initField(z_temp, &z_temp_, mesh_.VertexStride, kSize_, do_reshape);
    }

    void copy_pointers(::dawn::float_type* kh_smag_e, ::dawn::float_type* inv_dual_edge_length,
                       ::dawn::float_type* theta_v, ::dawn::float_type* z_temp) {
      kh_smag_e_ = kh_smag_e;
      inv_dual_edge_length_ = inv_dual_edge_length;
      theta_v_ = theta_v;
      z_temp_ = z_temp;
    }
  };
};
} // namespace cuda_ico
} // namespace dawn_generated
extern "C" {
void run_unroll_v_c_v_unroll_from_c_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                         ::dawn::float_type* kh_smag_e,
                                         ::dawn::float_type* inv_dual_edge_length,
                                         ::dawn::float_type* theta_v, ::dawn::float_type* z_temp) {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37 s;
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::setup(mesh, k_size, 0);
  s.copy_memory(kh_smag_e, inv_dual_edge_length, theta_v, z_temp, true);
  s.run();
  s.CopyResultToHost(z_temp, true);
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::free();
  return;
}
void run_unroll_v_c_v_unroll_from_fort_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                            ::dawn::float_type* kh_smag_e,
                                            ::dawn::float_type* inv_dual_edge_length,
                                            ::dawn::float_type* theta_v,
                                            ::dawn::float_type* z_temp) {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37 s;
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::setup(mesh, k_size, 0);
  s.copy_memory(kh_smag_e, inv_dual_edge_length, theta_v, z_temp, false);
  s.run();
  s.CopyResultToHost(z_temp, false);
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::free();
  return;
}
void run_unroll_v_c_v_unroll(::dawn::float_type* kh_smag_e,
                             ::dawn::float_type* inv_dual_edge_length, ::dawn::float_type* theta_v,
                             ::dawn::float_type* z_temp) {
  dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37 s;
  s.copy_pointers(kh_smag_e, inv_dual_edge_length, theta_v, z_temp);
  s.run();
  return;
}
bool verify_unroll_v_c_v_unroll(const ::dawn::float_type* z_temp_dsl,
                                const ::dawn::float_type* z_temp, const double z_temp_rel_tol,
                                const double z_temp_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto& mesh = dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::getMesh();
  int kSize = dawn_generated::cuda_ico::unroll_v_c_v_unroll::stencil_37::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  bool isValid;
  isValid = ::dawn::verify_field((mesh.VertexStride) * kSize, z_temp_dsl, z_temp, "z_temp",
                                 z_temp_rel_tol, z_temp_abs_tol);
  if(!isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_verts(0, (mesh.NumVertices - 1), kSize, (mesh.VertexStride), z_temp,
                          "unroll_v_c_v_unroll", "z_temp", iteration);
    serialize_dense_verts(0, (mesh.NumVertices - 1), kSize, (mesh.VertexStride), z_temp_dsl,
                          "unroll_v_c_v_unroll", "z_temp_dsl", iteration);
    std::cout << "[DSL] serializing z_temp as error is high.\n" << std::flush;
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
                                        ::dawn::float_type* theta_v, ::dawn::float_type* z_temp,
                                        ::dawn::float_type* z_temp_before,
                                        const double z_temp_rel_tol, const double z_temp_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil unroll_v_c_v_unroll (" << iteration << ") ...\n"
            << std::flush;
  run_unroll_v_c_v_unroll(kh_smag_e, inv_dual_edge_length, theta_v, z_temp_before);
  std::cout << "[DSL] unroll_v_c_v_unroll run time: " << time << "s\n" << std::flush;
  std::cout << "[DSL] Verifying stencil unroll_v_c_v_unroll...\n" << std::flush;
  verify_unroll_v_c_v_unroll(z_temp_before, z_temp, z_temp_rel_tol, z_temp_abs_tol, iteration);
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

