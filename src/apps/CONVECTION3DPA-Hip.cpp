//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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

#define CONVECTION3DPA_DATA_SETUP_HIP                                         \
  allocAndInitHipDeviceData(Basis, m_B, CPA_Q1D *CPA_D1D);                    \
  allocAndInitHipDeviceData(tBasis, m_Bt, CPA_Q1D *CPA_D1D);                  \
  allocAndInitHipDeviceData(dBasis, m_G, CPA_Q1D *CPA_D1D);                   \
  allocAndInitHipDeviceData(D, m_D, CPA_Q1D *CPA_Q1D *CPA_Q1D *CPA_VDIM *m_NE); \
  allocAndInitHipDeviceData(X, m_X, CPA_D1D *CPA_D1D *CPA_D1D *m_NE);         \
  allocAndInitHipDeviceData(Y, m_Y, CPA_D1D *CPA_D1D *CPA_D1D *m_NE);

#define CONVECTION3DPA_DATA_TEARDOWN_HIP                                       \
  getHipDeviceData(m_Y, Y, CPA_D1D *CPA_D1D *CPA_D1D *m_NE);                  \
  deallocHipDeviceData(Basis);                                                \
  deallocHipDeviceData(tBasis);                                               \
  deallocHipDeviceData(dBasis);                                               \
  deallocHipDeviceData(D);                                                    \
  deallocHipDeviceData(X);                                                    \
  deallocHipDeviceData(Y);

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

  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    CONVECTION3DPA_DATA_SETUP_HIP;

    dim3 nblocks(NE);
    dim3 nthreads_per_block(CPA_Q1D, CPA_Q1D, CPA_Q1D);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipLaunchKernelGGL((Convection3DPA<block_size>),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         Basis, tBasis, dBasis, D, X, Y);

      hipErrchk(hipGetLastError());
    }
    stopTimer();

    CONVECTION3DPA_DATA_TEARDOWN_HIP;

    break;
  }

  case RAJA_HIP: {

    CONVECTION3DPA_DATA_SETUP_HIP;

    constexpr bool async = true;

    using launch_policy =
        RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<async, CPA_Q1D*CPA_Q1D*CPA_Q1D>>;

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
                           RAJA::expt::Threads(CPA_Q1D, CPA_Q1D, CPA_Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

             CONVECTION3DPA_0_GPU;

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dy)
                      ); // RAJA::expt::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::expt::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qz)
                      ); // RAJA::expt::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::expt::loop<inner_x>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dz)
                      ); // RAJA::expt::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::expt::loop<inner_x>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::expt::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::expt::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

                          CONVECTION3DPA_8;

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

    CONVECTION3DPA_DATA_TEARDOWN_HIP;

    break;
  }

  default: {

    getCout() << "\n CONVECTION3DPA : Unknown Hip variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(CONVECTION3DPA, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
