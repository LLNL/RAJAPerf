//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DPA_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DPA_Atomic(const Real_ptr B,
                                const Real_ptr D, const Real_ptr X, const Index_ptr ElemToDoF, Real_ptr Y) {

  const Index_type e = blockIdx.x;

  MASS3DPA_ATOMIC_0_GPU;


  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        MASS3DPA_ATOMIC_1;
      }
    }
  }

  GPU_FOREACH_THREAD_DIRECT(dz, z, 1) {
    GPU_FOREACH_THREAD_DIRECT(d, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(q, x, mpa_at::Q1D) {
        MASS3DPA_ATOMIC_2;
      }
    }
  }
  __syncthreads();


  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
        MASS3DPA_ATOMIC_3;
      }
    }
  }
  __syncthreads();


  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
      MASS3DPA_ATOMIC_4;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
        MASS3DPA_ATOMIC_5;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
      MASS3DPA_ATOMIC_6;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        MASS3DPA_ATOMIC_7;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
      MASS3DPA_ATOMIC_8;
      MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_HIP);
      }
    }
  }

}

template < size_t block_size >
void MASS3DPA_ATOMIC::runHipVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  MASS3DPA_ATOMIC_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
      dim3 nthreads_per_block(mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (Mass3DPA_Atomic<block_size>),
                         NE, nthreads_per_block,
                         shmem, res.get_stream(),
                         B, D, X, ElemToDoF, Y );
      RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");

    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, mpa_at::Q1D*mpa_at::Q1D*mpa_at::Q1D>>;

    using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<mpa_at::Q1D>>;

    using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_size_y_direct<mpa_at::Q1D>>;

    using inner_z = RAJA::LoopPolicy<RAJA::hip_thread_size_z_direct<mpa_at::Q1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                         RAJA::Threads(mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {


            MASS3DPA_ATOMIC_0_GPU;

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_1;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
              [&](Index_type ) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type d) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type q) {
                      MASS3DPA_ATOMIC_2;
                      } // lambda (q)
                    ); // RAJA::loop<inner_x>
                  } // lambda (d)
                ); // RAJA::loop<inner_y>
              } // lambda ()
            ); // RAJA::loop<inner_z>
            ctx.teamSync();


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_3;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_4;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_5;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_6;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_7;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_8;
                      MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_RAJA_HIP);
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASS3DPA_ATOMIC : Unknown Hip variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DPA_ATOMIC, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
