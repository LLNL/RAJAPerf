//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Convection3DPA(const Real_ptr Basis, const Real_ptr tBasis,
                              const Real_ptr dBasis, const Real_ptr D,
                              const Real_ptr X, Real_ptr Y) {

  const int e = blockIdx.x;

  CONVECTION3DPA_0_GPU;

  GPU_FOREACH_THREAD(dz,z,CPA_D1D)
  {
    GPU_FOREACH_THREAD(dy,y,CPA_D1D)
    {
      GPU_FOREACH_THREAD(dx,x,CPA_D1D)
      {
        CONVECTION3DPA_1;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,CPA_D1D)
  {
    GPU_FOREACH_THREAD(dy,y,CPA_D1D)
    {
      GPU_FOREACH_THREAD(qx,x,CPA_Q1D)
      {
        CONVECTION3DPA_2;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,CPA_D1D)
  {
    GPU_FOREACH_THREAD(qx,x,CPA_Q1D)
    {
      GPU_FOREACH_THREAD(qy,y,CPA_Q1D)
      {
        CONVECTION3DPA_3;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(qx,x,CPA_Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,CPA_Q1D)
    {
      GPU_FOREACH_THREAD(qz,z,CPA_Q1D)
      {
        CONVECTION3DPA_4;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(qz,z,CPA_Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,CPA_Q1D)
    {
      GPU_FOREACH_THREAD(qx,x,CPA_Q1D)
      {
        CONVECTION3DPA_5;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(qx,x,CPA_Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,CPA_Q1D)
    {
      GPU_FOREACH_THREAD(dz,z,CPA_D1D)
      {
        CONVECTION3DPA_6;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,CPA_D1D)
  {
    GPU_FOREACH_THREAD(qx,x,CPA_Q1D)
    {
      GPU_FOREACH_THREAD(dy,y,CPA_D1D)
      {
        CONVECTION3DPA_7;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,CPA_D1D)
  {
    GPU_FOREACH_THREAD(dy,y,CPA_D1D)
    {
      GPU_FOREACH_THREAD(dx,x,CPA_D1D)
      {
        CONVECTION3DPA_8;
      }
    }
  }

}

template < size_t block_size >
void CONVECTION3DPA::runHipVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    dim3 nblocks(NE);
    dim3 nthreads_per_block(CPA_Q1D, CPA_Q1D, CPA_Q1D);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      constexpr size_t shmem = 0;
      hipLaunchKernelGGL((Convection3DPA<block_size>),
                         dim3(nblocks), dim3(nthreads_per_block), shmem, res.get_stream(),
                         Basis, tBasis, dBasis, D, X, Y);

      hipErrchk(hipGetLastError());
    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, CPA_Q1D*CPA_Q1D*CPA_Q1D>>;

    using outer_x =
        RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<CPA_Q1D>>;

    using inner_y =
        RAJA::LoopPolicy<RAJA::hip_thread_size_y_loop<CPA_Q1D>>;

    using inner_z =
        RAJA::LoopPolicy<RAJA::hip_thread_size_z_loop<CPA_Q1D>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(CPA_Q1D, CPA_Q1D, CPA_Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

             CONVECTION3DPA_0_GPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

                          CONVECTION3DPA_8;

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

    getCout() << "\n CONVECTION3DPA : Unknown Hip variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(CONVECTION3DPA, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
