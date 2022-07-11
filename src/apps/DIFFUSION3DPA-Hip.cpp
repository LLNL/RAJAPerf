//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define DIFFUSION3DPA_DATA_SETUP_HIP                                           \
  allocAndInitHipDeviceData(Basis, m_B, DPA_Q1D *DPA_D1D);                     \
  allocAndInitHipDeviceData(dBasis, m_G, DPA_Q1D *DPA_D1D);                    \
  allocAndInitHipDeviceData(D, m_D, DPA_Q1D *DPA_Q1D *DPA_Q1D *SYM *m_NE);     \
  allocAndInitHipDeviceData(X, m_X, DPA_D1D *DPA_D1D *DPA_D1D *m_NE);          \
  allocAndInitHipDeviceData(Y, m_Y, DPA_D1D *DPA_D1D *DPA_D1D *m_NE);

#define DIFFUSION3DPA_DATA_TEARDOWN_HIP                                        \
  getHipDeviceData(m_Y, Y, DPA_D1D *DPA_D1D *DPA_D1D *m_NE);                   \
  deallocHipDeviceData(Basis);                                                 \
  deallocHipDeviceData(dBasis);                                                \
  deallocHipDeviceData(D);                                                     \
  deallocHipDeviceData(X);                                                     \
  deallocHipDeviceData(Y);

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Diffusion3DPA(const Real_ptr Basis,
                              const Real_ptr dBasis, const Real_ptr D,
                              const Real_ptr X, Real_ptr Y, bool symmetric) {

  const int e = hipBlockIdx_x;

  DIFFUSION3DPA_0_GPU;

  GPU_FOREACH_THREAD(dz, z, DPA_D1D) {
    GPU_FOREACH_THREAD(dy, y, DPA_D1D) {
      GPU_FOREACH_THREAD(dx, x, DPA_D1D) {
        DIFFUSION3DPA_1;
      }
    }
  }

  if (threadIdx.z == 0) {
    GPU_FOREACH_THREAD(dy, y, DPA_D1D) {
      GPU_FOREACH_THREAD(qx, x, DPA_Q1D) {
        DIFFUSION3DPA_2;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(dz, z, DPA_D1D) {
    GPU_FOREACH_THREAD(dy, y, DPA_D1D) {
      GPU_FOREACH_THREAD(qx, x, DPA_Q1D) {
        DIFFUSION3DPA_3;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(dz, z, DPA_D1D) {
    GPU_FOREACH_THREAD(qy, y, DPA_Q1D) {
      GPU_FOREACH_THREAD(qx, x, DPA_Q1D) {
        DIFFUSION3DPA_4;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qz, z, DPA_Q1D) {
    GPU_FOREACH_THREAD(qy, y, DPA_Q1D) {
      GPU_FOREACH_THREAD(qx, x, DPA_Q1D) {
        DIFFUSION3DPA_5;
      }
    }
  }
  __syncthreads();
  if (threadIdx.z == 0) {
    GPU_FOREACH_THREAD(d, y, DPA_D1D) {
      GPU_FOREACH_THREAD(q, x, DPA_Q1D) {
        DIFFUSION3DPA_6;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qz, z, DPA_Q1D) {
    GPU_FOREACH_THREAD(qy, y, DPA_Q1D) {
      GPU_FOREACH_THREAD(dx, x, DPA_D1D) {
        DIFFUSION3DPA_7;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qz, z, DPA_Q1D) {
    GPU_FOREACH_THREAD(dy, y, DPA_D1D) {
      GPU_FOREACH_THREAD(dx, x, DPA_D1D) {
        DIFFUSION3DPA_8;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(dz, z, DPA_D1D) {
    GPU_FOREACH_THREAD(dy, y, DPA_D1D) {
      GPU_FOREACH_THREAD(dx, x, DPA_D1D) {
        DIFFUSION3DPA_9;
      }
    }
  }
}

template < size_t block_size >
void DIFFUSION3DPA::runHipVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    DIFFUSION3DPA_DATA_SETUP_HIP;

    dim3 nblocks(NE);
    dim3 nthreads_per_block(DPA_Q1D, DPA_Q1D, DPA_Q1D);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipLaunchKernelGGL((Diffusion3DPA<block_size>),
          dim3(nblocks), dim3(nthreads_per_block), 0, 0,
          Basis, dBasis, D, X, Y, symmetric);

      hipErrchk(hipGetLastError());
    }
    stopTimer();

    DIFFUSION3DPA_DATA_TEARDOWN_HIP;

    break;
  }

  case RAJA_HIP: {

    DIFFUSION3DPA_DATA_SETUP_HIP;

    constexpr bool async = true;

    using launch_policy =
        RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<async, DPA_Q1D*DPA_Q1D*DPA_Q1D>>;

    using outer_x =
        RAJA::expt::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x =
        RAJA::expt::LoopPolicy<RAJA::hip_thread_x_loop>;

    using inner_y =
        RAJA::expt::LoopPolicy<RAJA::hip_thread_y_loop>;

    using inner_z =
        RAJA::expt::LoopPolicy<RAJA::hip_thread_z_loop>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
          RAJA::expt::Grid(RAJA::expt::Teams(NE),
                           RAJA::expt::Threads(DPA_Q1D, DPA_Q1D, DPA_Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              DIFFUSION3DPA_0_GPU;

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                        [&](int dx) {

                          DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](int RAJA_UNUSED_ARG(dz)) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int qx) {

                         DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](int RAJA_UNUSED_ARG(dz)) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int d) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int q) {

                         DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
               [&](int dz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_9;

                       } // lambda (dx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::expt::loop<inner_z>

            } // lambda (e)
          ); // RAJA::expt::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::expt::launch

    } // loop over kernel reps
    stopTimer();

    DIFFUSION3DPA_DATA_TEARDOWN_HIP;

    break;
  }

  default: {

    getCout() << "\n DIFFUSION3DPA : Unknown Hip variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(DIFFUSION3DPA, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
