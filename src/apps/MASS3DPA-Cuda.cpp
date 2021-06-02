//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define MASS3DPA_DATA_SETUP_CUDA                                               \
  allocAndInitCudaDeviceData(B, m_B, m_Q1D *m_D1D);                            \
  allocAndInitCudaDeviceData(Bt, m_Bt, m_Q1D *m_D1D);                          \
  allocAndInitCudaDeviceData(D, m_D, m_Q1D *m_Q1D *m_Q1D *m_NE);               \
  allocAndInitCudaDeviceData(X, m_X, m_D1D *m_D1D *m_D1D *m_NE);               \
  allocAndInitCudaDeviceData(Y, m_Y, m_D1D *m_D1D *m_D1D *m_NE);

#define MASS3DPA_DATA_TEARDOWN_CUDA                                            \
  getCudaDeviceData(m_Y, Y, m_D1D *m_D1D *m_D1D *m_NE);                        \
  deallocCudaDeviceData(B);                                                    \
  deallocCudaDeviceData(Bt);                                                   \
  deallocCudaDeviceData(D);                                                    \
  deallocCudaDeviceData(X);                                                    \
  deallocCudaDeviceData(Y);

#define D1D 4
#define Q1D 5
#define B_(x, y) B[x + Q1D * y]
#define Bt_(x, y) Bt[x + D1D * y]
#define s_xy_(x, y) s_xy[x + M1D * y]
#define X_(dx, dy, dz, e)                                                      \
  X[dx + D1D * dy + D1D * D1D * dz + D1D * D1D * D1D * e]
#define Y_(dx, dy, dz, e)                                                      \
  Y[dx + D1D * dy + D1D * D1D * dz + D1D * D1D * D1D * e]
#define D_(qx, qy, qz, e)                                                      \
  D[qx + Q1D * qy + Q1D * Q1D * qz + Q1D * Q1D * Q1D * e]

#define RAJA_DIRECT_PRAGMA(X) _Pragma(#X)
#define RAJA_UNROLL(N) RAJA_DIRECT_PRAGMA(unroll(N))
#define FOREACH_THREAD(i, k, N)                                                \
  for (int i = threadIdx.k; i < N; i += blockDim.k)

__global__ void Mass3DPA(Index_type NE, const Real_ptr B, const Real_ptr Bt,
                         const Real_ptr D, const Real_ptr X, Real_ptr Y) {

  const int e = blockIdx.x;

  constexpr int MQ1 = Q1D;
  constexpr int MD1 = D1D;
  constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
  double sDQ[MQ1 * MD1];
  double(*Bsmem)[MD1] = (double(*)[MD1])sDQ;
  double(*Btsmem)[MQ1] = (double(*)[MQ1])sDQ;
  double sm0[MDQ * MDQ * MDQ];
  double sm1[MDQ * MDQ * MDQ];
  double(*Xsmem)[MD1][MD1] = (double(*)[MD1][MD1])sm0;
  double(*DDQ)[MD1][MQ1] = (double(*)[MD1][MQ1])sm1;
  double(*DQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm0;
  double(*QQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm1;
  double(*QQD)[MQ1][MD1] = (double(*)[MQ1][MD1])sm0;
  double(*QDD)[MD1][MD1] = (double(*)[MD1][MD1])sm1;

  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(dx, x, D1D) {
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; ++dz) {
        Xsmem[dz][dy][dx] = X_(dx, dy, dz, e);
      }
    }
    FOREACH_THREAD(dx, x, Q1D) { Bsmem[dx][dy] = B_(dx, dy); }
  }
  __syncthreads();
  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(qx, x, Q1D) {
      double u[D1D];
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; dz++) {
        u[dz] = 0;
      }
      RAJA_UNROLL(MD1)
      for (int dx = 0; dx < D1D; ++dx) {
        RAJA_UNROLL(MD1)
        for (int dz = 0; dz < D1D; ++dz) {
          u[dz] += Xsmem[dz][dy][dx] * Bsmem[qx][dx];
        }
      }
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; ++dz) {
        DDQ[dz][dy][qx] = u[dz];
      }
    }
  }
  __syncthreads();
  FOREACH_THREAD(qy, y, Q1D) {
    FOREACH_THREAD(qx, x, Q1D) {
      double u[D1D];
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; dz++) {
        u[dz] = 0;
      }
      RAJA_UNROLL(MD1)
      for (int dy = 0; dy < D1D; ++dy) {
        RAJA_UNROLL(MD1)
        for (int dz = 0; dz < D1D; dz++) {
          u[dz] += DDQ[dz][dy][qx] * Bsmem[qy][dy];
        }
      }
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; dz++) {
        DQQ[dz][qy][qx] = u[dz];
      }
    }
  }
  __syncthreads();
  FOREACH_THREAD(qy, y, Q1D) {
    FOREACH_THREAD(qx, x, Q1D) {
      double u[Q1D];

      RAJA_UNROLL(MQ1)
      for (int qz = 0; qz < Q1D; qz++) {
        u[qz] = 0;
      }
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; ++dz) {
        RAJA_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; qz++) {
          u[qz] += DQQ[dz][qy][qx] * Bsmem[qz][dz];
        }
      }
      RAJA_UNROLL(MQ1)
      for (int qz = 0; qz < Q1D; qz++) {
        QQQ[qz][qy][qx] = u[qz] * D_(qx, qy, qz, e);
      }
    }
  }

  __syncthreads();
  FOREACH_THREAD(d, y, D1D) {
    FOREACH_THREAD(q, x, Q1D) { Btsmem[d][q] = Bt_(q, d); }
  }

  __syncthreads();
  FOREACH_THREAD(qy, y, Q1D) {
    FOREACH_THREAD(dx, x, D1D) {
      double u[Q1D];
      RAJA_UNROLL(MQ1)
      for (int qz = 0; qz < Q1D; ++qz) {
        u[qz] = 0;
      }
      RAJA_UNROLL(MQ1)
      for (int qx = 0; qx < Q1D; ++qx) {
        RAJA_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; ++qz) {
          u[qz] += QQQ[qz][qy][qx] * Btsmem[dx][qx];
        }
      }
      RAJA_UNROLL(MQ1)
      for (int qz = 0; qz < Q1D; ++qz) {
        QQD[qz][qy][dx] = u[qz];
      }
    }
  }
  __syncthreads();

  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(dx, x, D1D) {
      double u[Q1D];
      RAJA_UNROLL(MQ1)
      for (int qz = 0; qz < Q1D; ++qz) {
        u[qz] = 0;
      }
      RAJA_UNROLL(MQ1)
      for (int qy = 0; qy < Q1D; ++qy) {
        RAJA_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; ++qz) {
          u[qz] += QQD[qz][qy][dx] * Btsmem[dy][qy];
        }
      }
      RAJA_UNROLL(MQ1)
      for (int qz = 0; qz < Q1D; ++qz) {
        QDD[qz][dy][dx] = u[qz];
      }
    }
  }

  __syncthreads();
  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(dx, x, D1D) {
      double u[D1D];
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; ++dz) {
        u[dz] = 0;
      }
      RAJA_UNROLL(MQ1)
      for (int qz = 0; qz < Q1D; ++qz) {
        RAJA_UNROLL(MD1)
        for (int dz = 0; dz < D1D; ++dz) {
          u[dz] += QDD[qz][dy][dx] * Btsmem[dz][qz];
        }
      }
      RAJA_UNROLL(MD1)
      for (int dz = 0; dz < D1D; ++dz) {
        Y_(dx, dy, dz, e) += u[dz];
      }
    }
  }
}

void MASS3DPA::runCudaVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_CUDA: {

    MASS3DPA_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(Q1D, Q1D, 1);

      Mass3DPA<<<NE, nthreads_per_block>>>(NE, B, Bt, D, X, Y);
    }
    stopTimer();

    MASS3DPA_DATA_TEARDOWN_CUDA;

    break;
  }

  case RAJA_CUDA: {

    break;
  }

  default: {

    std::cout << "\n MASS3DPA : Unknown Cuda variant id = " << vid << std::endl;
    break;
  }
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
