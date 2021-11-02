static void setup_cec(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                      cudaStream_t stream, const int kh_smag_e_kSize,
                      const int theta_v_kSize, const int exner_kSize) {
  mesh_ = GpuTriMesh(mesh);
  kSize_ = kSize;
  is_setup_ = true;
  stream_ = stream;
 
  int *ccTable_h = new int[C_C_SIZE * mesh_.CellStride];
  int *ecTable_h = new int[E_C_SIZE * mesh_.EdgeStride];
  int *ceTable_h = new int[C_E_SIZE * mesh_.CellStride];

  cudaMemcpy(ecTable_h, mesh_.ecTable,
             sizeof(int) * E_C_SIZE * mesh_.EdgeStride, cudaMemcpyDeviceToHost);
  cudaMemcpy(ceTable_h, mesh_.ceTable,
             sizeof(int) * C_E_SIZE * mesh_.CellStride, cudaMemcpyDeviceToHost);

  std::fill(ccTable_h, ccTable_h + mesh_.CellStride * C_C_SIZE, -1);

  for (int elemIdx = 0; elemIdx < mesh_.CellStride; elemIdx++) {
    for (int nbhIter0 = 0; nbhIter0 < C_E_SIZE; nbhIter0++) {
      int nbhIdx0 = ceTable_h[elemIdx + mesh_.CellStride * nbhIter0];
      if (nbhIdx0 == DEVICE_MISSING_VALUE) {
        continue;
      }
      if (ecTable_h[nbhIdx0 + mesh_.EdgeStride * 0] != elemIdx) {
        ccTable_h[elemIdx + mesh_.CellStride * nbhIter0] =
            ecTable_h[nbhIdx0 + mesh_.EdgeStride * 0];
      } else {
        ccTable_h[elemIdx + mesh_.CellStride * nbhIter0] =
            ecTable_h[nbhIdx0 + mesh_.EdgeStride * 1];
      }
    }
  }

  cudaMalloc((void **)&mesh_.ccTable,
             sizeof(int) * mesh_.CellStride * C_C_SIZE);
  cudaMemcpy(mesh_.ccTable, ccTable_h,
             sizeof(int) * mesh_.CellStride * C_C_SIZE, cudaMemcpyHostToDevice);
}

static void setup_vev(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                      cudaStream_t stream, const int kh_smag_e_kSize,
                      const int theta_v_kSize, const int exner_kSize) {
  mesh_ = GpuTriMesh(mesh);
  kSize_ = kSize;
  is_setup_ = true;
  stream_ = stream;

  int *vvTable_h = new int[V_V_SIZE * mesh_.VertexStride];
  int *evTable_h = new int[E_V_SIZE * mesh_.EdgeStride];
  int *veTable_h = new int[V_E_SIZE * mesh_.VertexStride];

  cudaMemcpy(evTable_h, mesh_.evTable,
             sizeof(int) * E_V_SIZE * mesh_.EdgeStride, cudaMemcpyDeviceToHost);
  cudaMemcpy(veTable_h, mesh_.veTable,
             sizeof(int) * V_E_SIZE * mesh_.VertexStride,
             cudaMemcpyDeviceToHost);

  std::fill(vvTable_h, vvTable_h + mesh_.VertexStride * V_V_SIZE, -1);

  for (int elemIdx = 0; elemIdx < mesh_.VertexStride; elemIdx++) {
    int lin_idx = 0;
    for (int nbhIter0 = 0; nbhIter0 < V_E_SIZE; nbhIter0++) {
      int nbhIdx0 = evTable_h[elemIdx + mesh_.VertexStride * nbhIter0];
      if (nbhIdx0 == DEVICE_MISSING_VALUE) {
        continue;
      }
      for (int nbhIter1 = 0; nbhIter1 < E_V_SIZE; nbhIter1++) {
        int nbhIdx1 = veTable_h[nbhIdx0 + mesh_.EdgeStride * nbhIter1];
        if (nbhIdx1 == DEVICE_MISSING_VALUE) {
          continue;
        }
        if (nbhIdx1 != nbhIdx0) {
          vvTable_h[elemIdx + mesh_.Vertex * lin_idx] = nbhIdx1;
          lin_idx++;
        }
      }
    }
  }

  cudaMalloc((void **)&mesh_.vvTable,
             sizeof(int) * mesh_.VertexStride * V_V_SIZE);
  cudaMemcpy(mesh_.vvTable, vvTable_h,
             sizeof(int) * mesh_.VertexStride * V_V_SIZE,
             cudaMemcpyHostToDevice);
}

static void setup_eve(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                      cudaStream_t stream, const int kh_smag_e_kSize,
                      const int theta_v_kSize, const int exner_kSize) {
  mesh_ = GpuTriMesh(mesh);
  kSize_ = kSize;
  is_setup_ = true;
  stream_ = stream;
  kh_smag_e_kSize_ = kh_smag_e_kSize;
  theta_v_kSize_ = theta_v_kSize;
  exner_kSize_ = exner_kSize;
  ::dawn::allocField(&enh_diffu_3d_, mesh_.EdgeStride, kSize_);
  ::dawn::allocField(&z_temp_, mesh_.EdgeStride, kSize_);

  int *eeTable_h = new int[E_E_SIZE * mesh_.EdgeStride];
  int *veTable_h = new int[V_E_SIZE * mesh_.EdgeStride];
  int *evTable_h = new int[E_V_SIZE * mesh_.VertexStride];

  cudaMemcpy(veTable_h, mesh_.veTable,
             sizeof(int) * V_E_SIZE * mesh_.EdgeStride, cudaMemcpyDeviceToHost);
  cudaMemcpy(evTable_h, mesh_.evTable,
             sizeof(int) * E_V_SIZE * mesh_.VertexStride,
             cudaMemcpyDeviceToHost);

  std::fill(eeTable_h, eeTable_h + mesh_.EdgeStride * E_E_SIZE, -1);

  for (int elemIdx = 0; elemIdx < mesh_.EdgeStride; elemIdx++) {
    int lin_idx = 0;
    for (int nbhIter0 = 0; nbhIter0 < V_E_SIZE; nbhIter0++) {
      int nbhIdx0 = evTable_h[elemIdx + mesh_.EdgeStride * nbhIter0];
      if (nbhIdx0 == DEVICE_MISSING_VALUE) {
        continue;
      }
      for (int nbhIter1 = 0; nbhIter1 < E_V_SIZE; nbhIter1++) {
        int nbhIdx1 = veTable_h[nbhIdx0 + mesh_.VertexStride * nbhIter1];
        if (nbhIdx1 == DEVICE_MISSING_VALUE) {
          continue;
        }
        if (nbhIdx1 != nbhIdx0) {
          eeTable_h[elemIdx + mesh_.EdgeStride * lin_idx] = nbhIdx1;
          lin_idx++;
        }
      }
    }
  }

  cudaMalloc((void **)&mesh_.eeTable,
             sizeof(int) * mesh_.EdgeStride * E_E_SIZE);
  cudaMemcpy(mesh_.eeTable, eeTable_h,
             sizeof(int) * mesh_.EdgeStride * E_E_SIZE, cudaMemcpyHostToDevice);
}

static void setup_ece(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                      cudaStream_t stream, const int kh_smag_e_kSize,
                      const int theta_v_kSize, const int exner_kSize) {
  mesh_ = GpuTriMesh(mesh);
  kSize_ = kSize;
  is_setup_ = true;
  stream_ = stream;
  kh_smag_e_kSize_ = kh_smag_e_kSize;
  theta_v_kSize_ = theta_v_kSize;
  exner_kSize_ = exner_kSize;
  ::dawn::allocField(&enh_diffu_3d_, mesh_.EdgeStride, kSize_);
  ::dawn::allocField(&z_temp_, mesh_.EdgeStride, kSize_);

  int *eeTable_h = new int[E_E_SIZE * mesh_.EdgeStride];
  int *ceTable_h = new int[C_E_SIZE * mesh_.EdgeStride];
  int *ecTable_h = new int[E_C_SIZE * mesh_.CellStride];

  cudaMemcpy(ceTable_h, mesh_.ceTable,
             sizeof(int) * C_E_SIZE * mesh_.EdgeStride, cudaMemcpyDeviceToHost);
  cudaMemcpy(ecTable_h, mesh_.ecTable,
             sizeof(int) * E_C_SIZE * mesh_.CellStride, cudaMemcpyDeviceToHost);

  std::fill(eeTable_h, eeTable_h + mesh_.EdgeStride * E_E_SIZE, -1);

  for (int elemIdx = 0; elemIdx < mesh_.EdgeStride; elemIdx++) {
    int lin_idx = 0;
    for (int nbhIter0 = 0; nbhIter0 < C_E_SIZE; nbhIter0++) {
      int nbhIdx0 = evTable_h[elemIdx + mesh_.EdgeStride * nbhIter0];
      if (nbhIdx0 == DEVICE_MISSING_VALUE) {
        continue;
      }
      for (int nbhIter1 = 0; nbhIter1 < E_C_SIZE; nbhIter1++) {
        int nbhIdx1 = veTable_h[nbhIdx0 + mesh_.CellStride * nbhIter1];
        if (nbhIdx1 == DEVICE_MISSING_VALUE) {
          continue;
        }
        if (nbhIdx1 != nbhIdx0) {
          eeTable_h[elemIdx + mesh_.EdgeStride * lin_idx] = nbhIdx1;
          lin_idx++;
        }
      }
    }
  }

  cudaMalloc((void **)&mesh_.eeTable,
             sizeof(int) * mesh_.EdgeStride * E_E_SIZE);
  cudaMemcpy(mesh_.eeTable, eeTable_h,
             sizeof(int) * mesh_.EdgeStride * E_E_SIZE, cudaMemcpyHostToDevice);
}

static void setup_cvc(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                      cudaStream_t stream, const int kh_smag_e_kSize,
                      const int theta_v_kSize, const int exner_kSize) {
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
    int lin_idx = 0;
    std::vector<std::set<int>> nbhs;
    for (int nbhIter0 = 0; nbhIter0 < C_V_SIZE; nbhIter0++) {
      std::<int>nbh;
      int nbhIdx0 = cvTable_h[elemIdx + mesh_.CellStride * nbhIter0];
      for (int nbhIter1 = 0; nbhIter1 < V_C_SIZE; nbhIter1++) {
        int nbhIdx1 = vcTable_h[nbhIdx0 + mesh_.VertexStride * nbhIter1];
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
          tour.insert(*it);
        }
      }
      tour.insert(anchor_right);

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

    assert(tour.size() == C_C_SIZE);
    for (int lin_idx = 0; lin_idx < C_C_SIZE; lin_idx) {
      ccTable_h[elemIdx + mesh_.CellStride * lin_idx] = tour[lin_idx];
    }
  }

  cudaMalloc((void **)&mesh_.ccTable,
             sizeof(int) * mesh_.CellStride * C_C_SIZE);
  cudaMemcpy(mesh_.ccTable, ccTable_h,
             sizeof(int) * mesh_.CellStride * C_C_SIZE, cudaMemcpyHostToDevice);
}

static void setup_vcv(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                      cudaStream_t stream, const int kh_smag_e_kSize,
                      const int theta_v_kSize, const int exner_kSize) {
  mesh_ = GpuTriMesh(mesh);
  kSize_ = kSize;
  is_setup_ = true;
  stream_ = stream;

  int *ccTable_h = new int[V_V_SIZE * mesh_.VertexStride];
  int *vcTable_h = new int[V_C_SIZE * mesh_.CellStride];
  int *cvTable_h = new int[C_V_SIZE * mesh_.VertexStride];

  cudaMemcpy(vcTable_h, mesh_.vcTable,
             sizeof(int) * V_C_SIZE * mesh_.CellStride, cudaMemcpyDeviceToHost);
  cudaMemcpy(cvTable_h, mesh_.cvTable,
             sizeof(int) * C_V_SIZE * mesh_.VertexStride,
             cudaMemcpyDeviceToHost);

  std::fill(vvTable_h, vvTable_h + mesh_.VertexStride * V_V_SIZE, -1);

  for (int elemIdx = 0; elemIdx < mesh_.VertexStride; elemIdx++) {
    int lin_idx = 0;
    std::vector<std::set<int>> nbhs;
    for (int nbhIter0 = 0; nbhIter0 < V_C_SIZE; nbhIter0++) {
      std::<int>nbh;
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
          tour.insert(*it);
        }
      }
      tour.insert(anchor_right);

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
    for (int lin_idx = 0; lin_idx < V_V_SIZE; lin_idx) {
      vvTable_h[elemIdx + mesh_.VertexStride * lin_idx] = tour[lin_idx];
    }
  }

  cudaMalloc((void **)&mesh_.vvTable,
             sizeof(int) * mesh_.VertexStride * V_V_SIZE);
  cudaMemcpy(mesh_.vvTable, vvTable_h,
             sizeof(int) * mesh_.VertexStride * V_V_SIZE,
             cudaMemcpyHostToDevice);
}