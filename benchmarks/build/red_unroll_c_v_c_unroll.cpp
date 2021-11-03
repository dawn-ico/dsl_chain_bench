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
cvc_kernel(int CellStride, int VertexStride, int kSize, int hOffset, int hSize,
           const int *cvTable, const int *ccTable,
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

    const int nbhIdx0_0 = cvTable[pidx + CellStride * 0];
    const int nbhIdx0_1 = cvTable[pidx + CellStride * 1];
    const int nbhIdx0_2 = cvTable[pidx + CellStride * 2];

    const int nbhIdx1_00 = kIter * CellStride + ccTable[pidx + CellStride * 0];
    const int nbhIdx1_01 = kIter * CellStride + ccTable[pidx + CellStride * 1];
    const int nbhIdx1_02 = kIter * CellStride + ccTable[pidx + CellStride * 2];
    const int nbhIdx1_03 = kIter * CellStride + ccTable[pidx + CellStride * 3];

    const int nbhIdx1_04 = kIter * CellStride + ccTable[pidx + CellStride * 4];
    const int nbhIdx1_05 = kIter * CellStride + ccTable[pidx + CellStride * 5];
    const int nbhIdx1_06 = kIter * CellStride + ccTable[pidx + CellStride * 6];
    const int nbhIdx1_07 = kIter * CellStride + ccTable[pidx + CellStride * 7];

    const int nbhIdx1_08 = kIter * CellStride + ccTable[pidx + CellStride * 8];
    const int nbhIdx1_09 = kIter * CellStride + ccTable[pidx + CellStride * 9];
    const int nbhIdx1_10 = kIter * CellStride + ccTable[pidx + CellStride * 10];
    const int nbhIdx1_11 = kIter * CellStride + ccTable[pidx + CellStride * 11];

    int self_idx = kIter * CellStride + pidx;

    ::dawn::float_type lhs_566 =
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_0] *
          inv_dual_edge_length[nbhIdx0_0]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_00] + theta_v[nbhIdx1_01] + theta_v[nbhIdx1_02] + theta_v[nbhIdx1_03] + theta_v[nbhIdx1_04])) + 
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_1] *
          inv_dual_edge_length[nbhIdx0_1]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_04] + theta_v[nbhIdx1_05] + theta_v[nbhIdx1_06] + theta_v[nbhIdx1_07] + theta_v[nbhIdx1_08])) + 
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_2] *
          inv_dual_edge_length[nbhIdx0_2]) *
         (theta_v[self_idx] + theta_v[nbhIdx1_08] + theta_v[nbhIdx1_09] + theta_v[nbhIdx1_10] + theta_v[nbhIdx1_11] + theta_v[nbhIdx1_00]));    

    z_temp[self_idx] = lhs_566;
  }
}

class unroll_c_v_c_unroll {
public:
  static const int C_V_SIZE = 3;
  static const int V_C_SIZE = 6;
  static const int C_C_SIZE = 12;

  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    dawn::unstructured_domain DomainLower;
    dawn::unstructured_domain DomainUpper;
    int* cvTable;
    int* vcTable;
    int* ccTable;

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
      cvTable = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{
          {dawn::LocationType::Cells, dawn::LocationType::Vertices}, 0});
      vcTable = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{
          {dawn::LocationType::Vertices, dawn::LocationType::Cells}, 0});
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
    
      int *ccTable_h = new int[C_C_SIZE * mesh_.CellStride];
      int *cvTable_h = new int[C_V_SIZE * mesh_.CellStride];
      int *vcTable_h = new int[V_C_SIZE * mesh_.VertexStride];

      cudaMemcpy(cvTable_h, mesh_.cvTable,
                sizeof(int) * C_V_SIZE * mesh_.CellStride, cudaMemcpyDeviceToHost);
      cudaMemcpy(vcTable_h, mesh_.vcTable,
                sizeof(int) * V_C_SIZE * mesh_.VertexStride,
                cudaMemcpyDeviceToHost);

      std::fill(ccTable_h, ccTable_h + mesh_.CellStride * C_C_SIZE, -1);
      
      for (int elemIdx = 0; elemIdx < mesh_.CellStride; elemIdx++) {       
        std::vector<std::set<int>> nbhs;
        for (int nbhIter0 = 0; nbhIter0 < C_V_SIZE; nbhIter0++) {
          std::set<int> nbh;
          int nbhIdx0 = cvTable_h[elemIdx + mesh_.CellStride * nbhIter0];
          for (int nbhIter1 = 0; nbhIter1 < V_C_SIZE; nbhIter1++) {
            int nbhIdx1 = vcTable_h[nbhIdx0 + mesh_.VertexStride * nbhIter1];
            if (nbhIdx1 != elemIdx) {
              nbh.insert(nbhIdx1);              
            }
          }
          // would like to put this but this isn't true
          //    there can be more than one -1 at the boundaries
          // assert(nbh.size() == 5); 
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

          // would like to put this but this isn't true
          //    two neighboring nbhs can intersect in a common value and a missing value
          // assert(intersect_right.size() == 1);
          // assert(intersect_left.size() == 1);

          int anchor_left = *intersect_left.begin();
          int anchor_right = *intersect_right.begin();
          
          tour.push_back(anchor_left);          
          for (auto it = nbhs[cur].begin(); it != nbhs[cur].end(); ++it) {
            if (*it != anchor_left && *it != anchor_right) {
              tour.push_back(*it);              
            }
          }          

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

        // would like to put this but this isn't true due to boundaries
        // assert(tour.size() == C_C_SIZE);                
        for (int lin_idx = 0; lin_idx < C_C_SIZE; lin_idx++) {          
          ccTable_h[elemIdx + mesh_.CellStride * lin_idx] = tour[lin_idx];
        }
      }     

      cudaMalloc((void **)&mesh_.ccTable,
                sizeof(int) * mesh_.CellStride * C_C_SIZE);
      cudaMemcpy(mesh_.ccTable, ccTable_h,
                sizeof(int) * mesh_.CellStride * C_C_SIZE, cudaMemcpyHostToDevice);
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
            "unroll_c_v_c_unroll has not been set up! make sure setup() is called before run!\n");
        return;
      }
      dim3 dB(BLOCK_SIZE, 1, 1);
      int hsize47 =
          mesh_.DomainUpper({::dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Halo, 0}) -
          mesh_.DomainLower(
              {::dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0}) +
          1;
      if(hsize47 == 0) {
        return;
      }
      int hoffset47 =
          mesh_.DomainLower({::dawn::LocationType::Cells, dawn::UnstructuredSubdomain::Nudging, 0});
      dim3 dG47 = grid(kSize_ + 0 - 0, hsize47, true);
      cvc_kernel<<<dG47, dB, 0, stream_>>>(
          mesh_.CellStride, mesh_.VertexStride, kSize_, hoffset47, hsize47, mesh_.cvTable,
          mesh_.ccTable, kh_smag_e_, inv_dual_edge_length_, theta_v_, z_temp_);
#ifndef NDEBUG

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    void CopyResultToHost(::dawn::float_type* z_temp, bool do_reshape) {
      if(do_reshape) {
        ::dawn::float_type* host_buf = new ::dawn::float_type[(mesh_.CellStride) * kSize_];
        gpuErrchk(cudaMemcpy((::dawn::float_type*)host_buf, z_temp_,
                             (mesh_.CellStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, z_temp, kSize_, mesh_.CellStride);
        delete[] host_buf;
      } else {
        gpuErrchk(cudaMemcpy(z_temp, z_temp_,
                             (mesh_.CellStride) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
    }

    void copy_memory(::dawn::float_type* kh_smag_e, ::dawn::float_type* inv_dual_edge_length,
                     ::dawn::float_type* theta_v, ::dawn::float_type* z_temp, bool do_reshape) {
      dawn::initField(kh_smag_e, &kh_smag_e_, mesh_.VertexStride, kSize_, do_reshape);
      dawn::initField(inv_dual_edge_length, &inv_dual_edge_length_, mesh_.VertexStride, 1,
                      do_reshape);
      dawn::initField(theta_v, &theta_v_, mesh_.CellStride, kSize_, do_reshape);
      dawn::initField(z_temp, &z_temp_, mesh_.CellStride, kSize_, do_reshape);
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
void run_unroll_c_v_c_unroll_from_c_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                         ::dawn::float_type* kh_smag_e,
                                         ::dawn::float_type* inv_dual_edge_length,
                                         ::dawn::float_type* theta_v, ::dawn::float_type* z_temp) {
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37 s;
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::setup(mesh, k_size, 0);
  s.copy_memory(kh_smag_e, inv_dual_edge_length, theta_v, z_temp, true);
  s.run();
  s.CopyResultToHost(z_temp, true);
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::free();
  return;
}
void run_unroll_c_v_c_unroll_from_fort_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                            ::dawn::float_type* kh_smag_e,
                                            ::dawn::float_type* inv_dual_edge_length,
                                            ::dawn::float_type* theta_v,
                                            ::dawn::float_type* z_temp) {
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37 s;
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::setup(mesh, k_size, 0);
  s.copy_memory(kh_smag_e, inv_dual_edge_length, theta_v, z_temp, false);
  s.run();
  s.CopyResultToHost(z_temp, false);
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::free();
  return;
}
void run_unroll_c_v_c_unroll(::dawn::float_type* kh_smag_e,
                             ::dawn::float_type* inv_dual_edge_length, ::dawn::float_type* theta_v,
                             ::dawn::float_type* z_temp) {
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37 s;
  s.copy_pointers(kh_smag_e, inv_dual_edge_length, theta_v, z_temp);
  s.run();
  return;
}
bool verify_unroll_c_v_c_unroll(const ::dawn::float_type* z_temp_dsl,
                                const ::dawn::float_type* z_temp, const double z_temp_rel_tol,
                                const double z_temp_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto& mesh = dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::getMesh();
  int kSize = dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  bool isValid;
  isValid = ::dawn::verify_field((mesh.CellStride) * kSize, z_temp_dsl, z_temp, "z_temp",
                                 z_temp_rel_tol, z_temp_abs_tol);
  if(!isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), kSize, (mesh.CellStride), z_temp,
                          "unroll_c_v_c_unroll", "z_temp", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), kSize, (mesh.CellStride), z_temp_dsl,
                          "unroll_c_v_c_unroll", "z_temp_dsl", iteration);
    std::cout << "[DSL] serializing z_temp as error is high.\n" << std::flush;
#endif
  }
#ifdef __SERIALIZE_ON_ERROR

      serialize_flush_iter("unroll_c_v_c_unroll", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n" << std::flush;
  return isValid;
}
void run_and_verify_unroll_c_v_c_unroll(::dawn::float_type* kh_smag_e,
                                        ::dawn::float_type* inv_dual_edge_length,
                                        ::dawn::float_type* theta_v, ::dawn::float_type* z_temp,
                                        ::dawn::float_type* z_temp_before,
                                        const double z_temp_rel_tol, const double z_temp_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil unroll_c_v_c_unroll (" << iteration << ") ...\n"
            << std::flush;
  run_unroll_c_v_c_unroll(kh_smag_e, inv_dual_edge_length, theta_v, z_temp_before);
  std::cout << "[DSL] unroll_c_v_c_unroll run time: " << time << "s\n" << std::flush;
  std::cout << "[DSL] Verifying stencil unroll_c_v_c_unroll...\n" << std::flush;
  verify_unroll_c_v_c_unroll(z_temp_before, z_temp, z_temp_rel_tol, z_temp_abs_tol, iteration);
  iteration++;
}
void setup_unroll_c_v_c_unroll(dawn::GlobalGpuTriMesh* mesh, int k_size, cudaStream_t stream) {
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::setup(mesh, k_size, stream);
}
void free_unroll_c_v_c_unroll() {
  dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::free();
}
}
int dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::kSize_;
cudaStream_t dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::stream_;
bool dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::is_setup_ = false;
dawn_generated::cuda_ico::unroll_c_v_c_unroll::GpuTriMesh
    dawn_generated::cuda_ico::unroll_c_v_c_unroll::stencil_37::mesh_;

