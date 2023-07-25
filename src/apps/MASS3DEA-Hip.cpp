//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DEA(const Real_ptr B, const Real_ptr D, Real_ptr M) {

  const int e = blockIdx.x;

  MASS3DEA_0

  GPU_FOREACH_THREAD(iz, z, 1) {
    GPU_FOREACH_THREAD(d, x, MEA_D1D) {
      GPU_FOREACH_THREAD(q, y, MEA_Q1D) {
        MASS3DEA_1
      }
    }
  }

  MASS3DEA_2

  GPU_FOREACH_THREAD(k1, x, MEA_Q1D) {
    GPU_FOREACH_THREAD(k2, y, MEA_Q1D) {
      GPU_FOREACH_THREAD(k3, z, MEA_Q1D) {
        MASS3DEA_3
      }
    }
  }

  __syncthreads();

  GPU_FOREACH_THREAD(i1, x, MEA_D1D) {
    GPU_FOREACH_THREAD(i2, y, MEA_D1D) {
      GPU_FOREACH_THREAD(i3, z, MEA_D1D) {
        MASS3DEA_4
      }
    }
  }
  
}

template < size_t block_size >
void MASS3DEA::runHipVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    dim3 nblocks(NE);
    dim3 nthreads_per_block(MEA_D1D, MEA_D1D, MEA_D1D);
    constexpr size_t shmem = 0;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipLaunchKernelGGL((Mass3DEA<block_size>), dim3(nblocks), dim3(nthreads_per_block), shmem, res.get_stream(),
                         B, D, M);

      hipErrchk( hipGetLastError() );
    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, MEA_D1D*MEA_D1D*MEA_D1D>>;

    using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<MEA_D1D>>;

    using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_size_y_loop<MEA_D1D>>;

    using inner_z = RAJA::LoopPolicy<RAJA::hip_thread_size_z_loop<MEA_D1D>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                         RAJA::Threads(MEA_D1D, MEA_D1D, MEA_D1D)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              MASS3DEA_0

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](int ) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                    [&](int d) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                        [&](int q) {
                          MASS3DEA_1
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_x>
                }
              ); // RAJA::loop<inner_z>


              MASS3DEA_2

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                [&](int k1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                    [&](int k2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                        [&](int k3) {
                          MASS3DEA_3
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                [&](int i1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                    [&](int i2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                        [&](int i3) {
                          MASS3DEA_4
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASS3DEA : Unknown Hip variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DEA, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
