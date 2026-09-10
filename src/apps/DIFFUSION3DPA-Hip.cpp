//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Diffusion3DPA(const Real_ptr Basis,
                              const Real_ptr dBasis, const Real_ptr D,
                              const Real_ptr X, Real_ptr Y, bool symmetric) {

  const Index_type e = blockIdx.x;

  DIFFUSION3DPA_0_GPU;

  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        DIFFUSION3DPA_1;
      }
    }
  }

  if (threadIdx.z == 0) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        DIFFUSION3DPA_2;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        DIFFUSION3DPA_3;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, diff::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        DIFFUSION3DPA_4;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD_DIRECT(qz, z, diff::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, diff::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        DIFFUSION3DPA_5;
      }
    }
  }
  __syncthreads();
  if (threadIdx.z == 0) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        DIFFUSION3DPA_6;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD_DIRECT(qz, z, diff::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, diff::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        DIFFUSION3DPA_7;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD_DIRECT(qz, z, diff::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        DIFFUSION3DPA_8;
      }
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        DIFFUSION3DPA_9;
      }
    }
  }

}

template < size_t block_size >
void DIFFUSION3DPA::runHipVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
      dim3 nthreads_per_block(diff::Q1D, diff::Q1D, diff::Q1D);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (Diffusion3DPA<block_size>),
                         NE, nthreads_per_block,
                         shmem, res.get_stream(),
                         Basis, dBasis, D, X, Y, symmetric );
      RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");
    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, diff::Q1D*diff::Q1D*diff::Q1D>>;

    using outer_x =
        RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<diff::Q1D>>;

    using inner_y =
        RAJA::LoopPolicy<RAJA::hip_thread_size_y_direct<diff::Q1D>>;

    using inner_z =
        RAJA::LoopPolicy<RAJA::hip_thread_size_z_direct<diff::Q1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(diff::Q1D, diff::Q1D, diff::Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              DIFFUSION3DPA_0_GPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                        [&](Index_type dx) {

                          DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type RAJA_UNUSED_ARG(dz)) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                          DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](Index_type RAJA_UNUSED_ARG(dz)) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                         DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
               [&](Index_type dz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

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
      //clang-format on
      RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");

    } // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n DIFFUSION3DPA : Unknown Hip variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DIFFUSION3DPA, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
