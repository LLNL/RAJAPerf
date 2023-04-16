//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Diffusion3DPA(const Real_ptr Basis,
                              const Real_ptr dBasis, const Real_ptr D,
                              const Real_ptr X, Real_ptr Y, bool symmetric) {

  const int e = blockIdx.x;

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
void DIFFUSION3DPA::runCudaVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_CUDA: {

    dim3 nthreads_per_block(DPA_Q1D, DPA_Q1D, DPA_Q1D);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Diffusion3DPA<block_size><<<NE, nthreads_per_block>>>(
          Basis, dBasis, D, X, Y, symmetric);

      cudaErrchk(cudaGetLastError());
    }
    stopTimer();

    break;
  }

  case RAJA_CUDA: {

    constexpr bool async = true;

    using launch_policy =
        RAJA::LaunchPolicy<RAJA::cuda_launch_t<async, DPA_Q1D*DPA_Q1D*DPA_Q1D>>;

    using outer_x =
        RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

    using inner_x =
        RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;

    using inner_y =
        RAJA::LoopPolicy<RAJA::cuda_thread_y_loop>;

    using inner_z =
        RAJA::LoopPolicy<RAJA::cuda_thread_z_loop>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>(
          RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(DPA_Q1D, DPA_Q1D, DPA_Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              DIFFUSION3DPA_0_GPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                        [&](int dx) {

                          DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](int RAJA_UNUSED_ARG(dz)) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int qx) {

                         DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](int RAJA_UNUSED_ARG(dz)) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int d) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int q) {

                         DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
               [&](int dz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_9;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

            } // lambda (e)
          ); // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch

    } // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n DIFFUSION3DPA : Unknown Cuda variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(DIFFUSION3DPA, Cuda)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
