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

  MASS3DPA_0_GPU

  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(dx, x, D1D){MASS3DPA_1} FOREACH_THREAD(dx, x, Q1D) {
      MASS3DPA_2
    }
  }
  __syncthreads();
  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(qx, x, Q1D) { MASS3DPA_3 }
  }
  __syncthreads();
  FOREACH_THREAD(qy, y, Q1D) {
    FOREACH_THREAD(qx, x, Q1D) { MASS3DPA_4 }
  }
  __syncthreads();
  FOREACH_THREAD(qy, y, Q1D) {
    FOREACH_THREAD(qx, x, Q1D) { MASS3DPA_5 }
  }

  __syncthreads();
  FOREACH_THREAD(d, y, D1D) {
    FOREACH_THREAD(q, x, Q1D) { MASS3DPA_6 }
  }

  __syncthreads();
  FOREACH_THREAD(qy, y, Q1D) {
    FOREACH_THREAD(dx, x, D1D) { MASS3DPA_7 }
  }
  __syncthreads();

  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(dx, x, D1D) { MASS3DPA_8 }
  }

  __syncthreads();
  FOREACH_THREAD(dy, y, D1D) {
    FOREACH_THREAD(dx, x, D1D) { MASS3DPA_9 }
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

    MASS3DPA_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
          RAJA::expt::DEVICE,
          RAJA::expt::Resources(RAJA::expt::Teams(NE),
                                RAJA::expt::Threads(Q1D, Q1D, 1)),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
            RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, NE), [&](int e) {

                  MASS3DPA_0_GPU

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                          MASS3DPA_1
                       });

                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, Q1D), [&](int dx) {
                          MASS3DPA_2
                      });
                   });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, Q1D), [&](int qx) {
                          MASS3DPA_3
                      });
                   });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, Q1D), [&](int qy) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, Q1D), [&](int qx) {
                          MASS3DPA_4
                      });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, Q1D), [&](int qy) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, Q1D), [&](int qx) {
                          MASS3DPA_5
                      });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, D1D), [&](int d) {
                    RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, Q1D), [&](int q) {
                        MASS3DPA_6
                     });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, Q1D), [&](int qy) {
                    RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                        MASS3DPA_7
                     });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                          MASS3DPA_8
                      });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                    RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                        MASS3DPA_9
                    });
                  });
              });
          });
    }
    stopTimer();

    MASS3DPA_DATA_TEARDOWN_CUDA;

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
