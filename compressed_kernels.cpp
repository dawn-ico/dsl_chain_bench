template <int E_C_SIZE, int C_E_SIZE>
__global__ void
cec_kernel(int CellStride, int EdgeStride, int kSize, int hOffset, int hSize,
           const int *ceTable, const int *ccTable,
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

    const int nbhIdx0_0 = ceTable[pidx + CellStride * 0];
    const int nbhIdx0_1 = ceTable[pidx + CellStride * 1];
    const int nbhIdx0_2 = ceTable[pidx + CellStride * 2];

    const int nbhIdx1_0 = ccTable[pidx + CellStride * 0];
    const int nbhIdx1_1 = ccTable[pidx + CellStride * 1];
    const int nbhIdx1_2 = ccTable[pidx + CellStride * 2];

    const int self_idx = kIter * CellStride + pidx;

    ::dawn::float_type lhs_566 = ((kh_smag_e[kIter * EdgeStride + nbhIdx0_0] *
                                   inv_dual_edge_length[nbhIdx0_0]) *
                                  (theta_v[self_idx] + theta_v[nbhIdx1_0])) +
                                 ((kh_smag_e[kIter * EdgeStride + nbhIdx0_1] *
                                   inv_dual_edge_length[nbhIdx0_1]) *
                                  (theta_v[self_idx] + theta_v[nbhIdx1_1])) +
                                 ((kh_smag_e[kIter * EdgeStride + nbhIdx0_2] *
                                   inv_dual_edge_length[nbhIdx0_2]) *
                                  (theta_v[self_idx] + theta_v[nbhIdx1_2]));

    z_temp[self_idx] = lhs_566;
  }
}

template <int E_V_SIZE, int V_E_SIZE>
__global__ void
vev_kernel(int VertexStride, int EdgeStride, int kSize, int hOffset, int hSize,
           const int *veTable, const int *vvTable,
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

    const int nbhIdx0_0 = veTable[pidx + VertexStride * 0];
    const int nbhIdx0_1 = veTable[pidx + VertexStride * 1];
    const int nbhIdx0_2 = veTable[pidx + VertexStride * 2];
    const int nbhIdx0_3 = veTable[pidx + VertexStride * 3];
    const int nbhIdx0_4 = veTable[pidx + VertexStride * 4];
    const int nbhIdx0_5 = veTable[pidx + VertexStride * 5];

    const int nbhIdx1_0 = vvTable[pidx + VertexStride * 0];
    const int nbhIdx1_1 = vvTable[pidx + VertexStride * 1];
    const int nbhIdx1_2 = vvTable[pidx + VertexStride * 2];
    const int nbhIdx1_3 = vvTable[pidx + VertexStride * 3];
    const int nbhIdx1_4 = vvTable[pidx + VertexStride * 4];
    const int nbhIdx1_5 = vvTable[pidx + VertexStride * 5];

    const int self_idx = kIter * VertexStride + pidx;

    ::dawn::float_type lhs_566 = ((kh_smag_e[kIter * EdgeStride + nbhIdx0_0] *
                                   inv_dual_edge_length[nbhIdx0_0]) *
                                  (theta_v[self] + theta_v[nbhIdx1_0])) +
                                 ((kh_smag_e[kIter * EdgeStride + nbhIdx0_1] *
                                   inv_dual_edge_length[nbhIdx0_1]) *
                                  (theta_v[self] + theta_v[nbhIdx1_1])) +
                                 ((kh_smag_e[kIter * EdgeStride + nbhIdx0_2] *
                                   inv_dual_edge_length[nbhIdx0_2]) *
                                  (theta_v[self] + theta_v[nbhIdx1_2])) +
                                 ((kh_smag_e[kIter * EdgeStride + nbhIdx0_3] *
                                   inv_dual_edge_length[nbhIdx0_3]) *
                                  (theta_v[self] + theta_v[nbhIdx1_3])) +
                                 ((kh_smag_e[kIter * EdgeStride + nbhIdx0_4] *
                                   inv_dual_edge_length[nbhIdx0_4]) *
                                  (theta_v[self] + theta_v[nbhIdx1_4])) +
                                 ((kh_smag_e[kIter * EdgeStride + nbhIdx0_5] *
                                   inv_dual_edge_length[nbhIdx0_5]) *
                                  (theta_v[self] + theta_v[nbhIdx1_5]));

    z_temp[self_idx] = lhs_566;
  }
}

template <int V_E_SIZE, int E_V_SIZE>
__global__ void
eve_kernel(int EdgeStride, int VertexStride, int kSize, int hOffset, int hSize,
           const int *evTable, const int *eeTable,
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

    const int nbhIdx0_0 = evTable[pidx + EdgeStride * 0];
    const int nbhIdx0_1 = evTable[pidx + EdgeStride * 1];

    const int nbhIdx1_0 = eeTable[pidx + EdgeStride * 0];
    const int nbhIdx1_1 = eeTable[pidx + EdgeStride * 1];
    const int nbhIdx1_2 = eeTable[pidx + EdgeStride * 2];
    const int nbhIdx1_3 = eeTable[pidx + EdgeStride * 3];
    const int nbhIdx1_4 = eeTable[pidx + EdgeStride * 4];
    const int nbhIdx1_5 = eeTable[pidx + EdgeStride * 5];
    const int nbhIdx1_6 = eeTable[pidx + EdgeStride * 6];
    const int nbhIdx1_7 = eeTable[pidx + EdgeStride * 7];
    const int nbhIdx1_8 = eeTable[pidx + EdgeStride * 8];
    const int nbhIdx1_9 = eeTable[pidx + EdgeStride * 9];

    int self_idx = kIter * EdgeStride + pidx;

    ::dawn::float_type lhs_566 =
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_0] *
          inv_dual_edge_length[nbhIdx0_0]) *
         (theta_v[self] + theta_v[nbhIdx1_0] + theta_v[nbhIdx1_1] +
          theta_v[nbhIdx1_2] + theta_v[nbhIdx1_3] + theta_v[nbhIdx1_4])) +
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_1] *
          inv_dual_edge_length[nbhIdx0_1]) *
         (theta_v[self] + theta_v[nbhIdx1_5] + theta_v[nbhIdx1_6] +
          theta_v[nbhIdx1_7] + theta_v[nbhIdx1_8] + theta_v[nbhIdx1_9]));

    z_temp[self_idx] = lhs_566;
  }
}

template <int C_E_SIZE, int E_C_SIZE>
__global__ void
ece_kernel(int EdgeStride, int CellStride, int kSize, int hOffset, int hSize,
           const int *ecTable, const int *eeTable,
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

    const int nbhIdx0_0 = ecTable[pidx + EdgeStride * 0];
    const int nbhIdx0_1 = ecTable[pidx + EdgeStride * 1];

    const int nbhIdx1_0 = eeTable[pidx + EdgeStride * 0];
    const int nbhIdx1_1 = eeTable[pidx + EdgeStride * 1];
    const int nbhIdx1_2 = eeTable[pidx + EdgeStride * 2];
    const int nbhIdx1_3 = eeTable[pidx + EdgeStride * 3];

    int self_idx = kIter * EdgeStride + pidx;

    ::dawn::float_type lhs_566 =
        ((kh_smag_e[kIter * CellStride + nbhIdx0_0] *
          inv_dual_edge_length[nbhIdx0_0]) *
         (theta_v[self] + theta_v[nbhIdx1_0] + theta_v[nbhIdx1_1]) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_1] *
          inv_dual_edge_length[nbhIdx0_1]) *
         (theta_v[self] + theta_v[nbhIdx1_2] + theta_v[nbhIdx1_3]));

    z_temp[self_idx] = lhs_566;
  }
}

template <int V_C_SIZE, int C_V_SIZE>
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

    const int nbhIdx1_00 = ccTable[pidx + CellStride * 0];
    const int nbhIdx1_01 = ccTable[pidx + CellStride * 1];
    const int nbhIdx1_02 = ccTable[pidx + CellStride * 2];
    const int nbhIdx1_03 = ccTable[pidx + CellStride * 3];

    const int nbhIdx1_04 = ccTable[pidx + CellStride * 4];
    const int nbhIdx1_05 = ccTable[pidx + CellStride * 5];
    const int nbhIdx1_06 = ccTable[pidx + CellStride * 6];
    const int nbhIdx1_07 = ccTable[pidx + CellStride * 7];

    const int nbhIdx1_08 = ccTable[pidx + CellStride * 8];
    const int nbhIdx1_09 = ccTable[pidx + CellStride * 9];
    const int nbhIdx1_10 = ccTable[pidx + CellStride * 10];
    const int nbhIdx1_11 = ccTable[pidx + CellStride * 11];

    int self_idx = kIter * CellStride + pidx;

    ::dawn::float_type lhs_566 =
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_0] *
          inv_dual_edge_length[nbhIdx0_0]) *
         (theta_v[self] + theta_v[nbhIdx1_00] + theta_v[nbhIdx1_01] + theta_v[nbhIdx1_02] + theta_v[nbhIdx1_03] + theta_v[nbhIdx1_04]) + 
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_1] *
          inv_dual_edge_length[nbhIdx0_1]) *
         (theta_v[self] + theta_v[nbhIdx1_04] + theta_v[nbhIdx1_05] + theta_v[nbhIdx1_06] + theta_v[nbhIdx1_07] + theta_v[nbhIdx1_08]) + 
        ((kh_smag_e[kIter * VertexStride + nbhIdx0_2] *
          inv_dual_edge_length[nbhIdx0_2]) *
         (theta_v[self] + theta_v[nbhIdx1_08] + theta_v[nbhIdx1_09] + theta_v[nbhIdx1_10] + theta_v[nbhIdx1_11] + theta_v[nbhIdx1_00]);

    z_temp[self_idx] = lhs_566;
  }
}

template <int C_V_SIZE, int V_C_SIZE>
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
         (theta_v[self] + theta_v[nbhIdx1_0] + theta_v[nbhIdx1_1])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_1] *
          inv_dual_edge_length[nbhIdx0_1]) *
         (theta_v[self] + theta_v[nbhIdx1_1] + theta_v[nbhIdx1_2])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_2] *
          inv_dual_edge_length[nbhIdx0_2]) *
         (theta_v[self] + theta_v[nbhIdx1_2] + theta_v[nbhIdx1_3])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_3] *
          inv_dual_edge_length[nbhIdx0_3]) *
         (theta_v[self] + theta_v[nbhIdx1_3] + theta_v[nbhIdx1_4])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_4] *
          inv_dual_edge_length[nbhIdx0_4]) *
         (theta_v[self] + theta_v[nbhIdx1_4] + theta_v[nbhIdx1_5])) +
        ((kh_smag_e[kIter * CellStride + nbhIdx0_5] *
          inv_dual_edge_length[nbhIdx0_5]) *
         (theta_v[self] + theta_v[nbhIdx1_5] + theta_v[nbhIdx1_0]));

    z_temp[self_idx] = lhs_566;
  }
}