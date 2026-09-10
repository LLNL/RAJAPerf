//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASSVEC3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define MASSVEC3DPA_GPU_KERNEL_BODY(SHARED_2D, SHARED_3D)              \
  {                                                                    \
    GPU_SHARED_2D_APPLY(SHARED_2D, q, d, mvpa::Q1D, mvpa::D1D) {       \
      MASSVEC3DPA_1;                                                   \
    }                                                                  \
                                                                       \
    for (Index_type c = 0; c < 3; ++c) {                               \
      GPU_SHARED_3D_APPLY(SHARED_3D, dx, dy, dz, mvpa::D1D, mvpa::D1D, \
                          mvpa::D1D)                                   \
      {                                                                \
        MASSVEC3DPA_2;                                                 \
      }                                                                \
      __syncthreads();                                                 \
                                                                       \
      GPU_SHARED_3D_APPLY(SHARED_3D, qx, dy, dz, mvpa::Q1D, mvpa::D1D, \
                          mvpa::D1D)                                   \
      {                                                                \
        MASSVEC3DPA_3;                                                 \
      }                                                                \
      __syncthreads();                                                 \
                                                                       \
      GPU_SHARED_3D_APPLY(SHARED_3D, qx, qy, dz, mvpa::Q1D, mvpa::Q1D, \
                          mvpa::D1D)                                   \
      {                                                                \
        MASSVEC3DPA_4;                                                 \
      }                                                                \
      __syncthreads();                                                 \
                                                                       \
      GPU_SHARED_3D_APPLY(SHARED_3D, qx, qy, qz, mvpa::Q1D, mvpa::Q1D, \
                          mvpa::Q1D)                                   \
      {                                                                \
        MASSVEC3DPA_5;                                                 \
      }                                                                \
      __syncthreads();                                                 \
                                                                       \
      GPU_SHARED_3D_APPLY(SHARED_3D, dx, qy, qz, mvpa::D1D, mvpa::Q1D, \
                          mvpa::Q1D)                                   \
      {                                                                \
        MASSVEC3DPA_6;                                                 \
      }                                                                \
      __syncthreads();                                                 \
                                                                       \
      GPU_SHARED_3D_APPLY(SHARED_3D, dx, dy, qz, mvpa::D1D, mvpa::D1D, \
                          mvpa::Q1D)                                   \
      {                                                                \
        MASSVEC3DPA_7;                                                 \
      }                                                                \
      __syncthreads();                                                 \
                                                                       \
      GPU_SHARED_3D_APPLY(SHARED_3D, dx, dy, dz, mvpa::D1D, mvpa::D1D, \
                          mvpa::D1D)                                   \
      {                                                                \
        MASSVEC3DPA_8;                                                 \
      }                                                                \
      __syncthreads();                                                 \
                                                                       \
    } /* (c) dimension loop */                                         \
	  }

template <size_t block_size>
__launch_bounds__(block_size) __global__ void MassVec3DPADirect(const Real_ptr B,
                                                                const Real_ptr D,
                                                                const Real_ptr X,
                                                                Real_ptr Y)
{

  const Index_type e = blockIdx.x;

  MASSVEC3DPA_0_GPU;

  MASSVEC3DPA_GPU_KERNEL_BODY(GPU_SHARED_DIRECT_2D, GPU_SHARED_DIRECT_3D);

}

template <size_t block_size>
__launch_bounds__(block_size) __global__ void MassVec3DPALoop(const Real_ptr B,
                                                              const Real_ptr D,
                                                              const Real_ptr X,
                                                              Real_ptr Y)
{

  const Index_type e = blockIdx.x;

  MASSVEC3DPA_0_GPU;

  MASSVEC3DPA_GPU_KERNEL_BODY(GPU_SHARED_LOOP_2D, GPU_SHARED_LOOP_3D);

}

#define MASSVEC3DPA_HIP_RAJA_LAUNCH                                            \
  {                                                                            \
    /* clang-format off */                                                     \
    RAJA::launch<launch_policy>(                                               \
      res,                                                                     \
      RAJA::LaunchParams(RAJA::Teams(NE),                                      \
      RAJA::Threads(mvpa::Q1D, mvpa::Q1D, mvpa::Q1D)),                         \
      [=] RAJA_HOST_DEVICE(launch_context ctx) {                               \
                                                                               \
        RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),                    \
          [&](Index_type e) {                                                  \
                                                                               \
            MASSVEC3DPA_0_GPU                                                  \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),                 \
              [&](Index_type) {                                                \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),     \
                  [&](Index_type d) {                                          \
                                                                               \
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),   \
                    [&](Index_type q) {                                        \
                      MASSVEC3DPA_1;                                           \
                    } /* lambda (q) */                                         \
                  ); /* RAJA::loop<inner_x> */                                 \
                  } /* lambda (d) */                                           \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda () */                                                \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            for (Index_type c = 0; c < 3; ++c) {                               \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),         \
              [&](Index_type dz) {                                             \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),     \
                  [&](Index_type dy) {                                         \
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), \
                      [&](Index_type dx) {                                     \
                        MASSVEC3DPA_2;                                         \
                      } /* lambda (dx) */                                      \
                    ); /* RAJA::loop<inner_x> */                               \
                  } /* lambda (dy) */                                          \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda (dz) */                                              \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            ctx.teamSync();                                                    \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),         \
              [&](Index_type dz) {                                             \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),     \
                  [&](Index_type dy) {                                         \
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), \
                      [&](Index_type qx) {                                     \
                        MASSVEC3DPA_3;                                         \
                      } /* lambda (qx) */                                      \
                    ); /* RAJA::loop<inner_x> */                               \
                  } /* lambda (dy) */                                          \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda (dz) */                                              \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            ctx.teamSync();                                                    \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),         \
              [&](Index_type dz) {                                             \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),     \
                  [&](Index_type qy) {                                         \
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), \
                      [&](Index_type qx) {                                     \
                        MASSVEC3DPA_4;                                         \
                      } /* lambda (qx) */                                      \
                    ); /* RAJA::loop<inner_x> */                               \
                  } /* lambda (qy) */                                          \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda (dz) */                                              \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            ctx.teamSync();                                                    \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),         \
              [&](Index_type qz) {                                             \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),     \
                  [&](Index_type qy) {                                         \
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), \
                    [&](Index_type qx) {                                       \
                    MASSVEC3DPA_5;                                             \
                    } /* lambda (qx) */                                        \
                    ); /* RAJA::loop<inner_x> */                               \
                  } /* lambda (qy) */                                          \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda (qz) */                                              \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            ctx.teamSync();                                                    \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),         \
              [&](Index_type qz) {                                             \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),     \
                  [&](Index_type qy) {                                         \
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), \
                      [&](Index_type dx) {                                     \
                        MASSVEC3DPA_6;                                         \
                      } /* lambda (dx) */                                      \
                    ); /* RAJA::loop<inner_x> */                               \
                  } /* lambda (qy) */                                          \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda (qz) */                                              \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            ctx.teamSync();                                                    \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),         \
              [&](Index_type qz) {                                             \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),     \
                  [&](Index_type dy) {                                         \
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), \
                      [&](Index_type dx) {                                     \
                      MASSVEC3DPA_7;                                           \
                      } /* lambda (dx) */                                      \
                    ); /* RAJA::loop<inner_x> */                               \
                  } /* lambda (dy) */                                          \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda (qz) */                                              \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            ctx.teamSync();                                                    \
                                                                               \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),         \
              [&](Index_type dz) {                                             \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),     \
                  [&](Index_type dy) {                                         \
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), \
                    [&](Index_type dx) {                                       \
                    MASSVEC3DPA_8;                                             \
                    } /* lambda (dx) */                                        \
                    ); /* RAJA::loop<inner_x> */                               \
                  } /* lambda (dy) */                                          \
                ); /* RAJA::loop<inner_y> */                                   \
              } /* lambda (dz) */                                              \
            ); /* RAJA::loop<inner_z> */                                       \
                                                                               \
            ctx.teamSync();                                                    \
                                                                               \
            } /* c - dim loop */                                               \
          } /* lambda (e) */                                                   \
        ); /* RAJA::loop<outer_x> */                                           \
      } /* outer lambda (ctx) */                                               \
    ); /* RAJA::launch */                                                      \
    /* clang-format on */                                                      \
                                                                               \
  }

template <size_t block_size, size_t tune_idx>
void MASSVEC3DPA::runHipVariantImpl(VariantID vid)
{

  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  MASSVEC3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    if constexpr (tune_idx == 0) {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
        dim3 nthreads_per_block(mvpa::Q1D, mvpa::Q1D, mvpa::Q1D);
        constexpr size_t shmem = 0;

        RPlaunchHipKernel((MassVec3DPALoop<block_size>), NE, nthreads_per_block,
                          shmem, res.get_stream(), B, D, X, Y);
        RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");
      }
      stopTimer();

    } else if constexpr (tune_idx == 1) {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
        dim3 nthreads_per_block(mvpa::Q1D, mvpa::Q1D, mvpa::Q1D);
        constexpr size_t shmem = 0;

        RPlaunchHipKernel((MassVec3DPADirect<block_size>), NE, nthreads_per_block,
                          shmem, res.get_stream(), B, D, X, Y);
        RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");
      }
      stopTimer();
    }

    break;
  }

  case RAJA_HIP: {

    if constexpr (tune_idx == 0) {

      constexpr bool async = true;

      using launch_policy = RAJA::LaunchPolicy<
      RAJA::hip_launch_t<async, mvpa::Q1D * mvpa::Q1D * mvpa::Q1D>>;

      using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

      using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_x_loop>;

      using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_y_loop>;

      using inner_z = RAJA::LoopPolicy<RAJA::hip_thread_z_loop>;

      using launch_context = RAJA::LaunchContext;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
        MASSVEC3DPA_HIP_RAJA_LAUNCH;
        RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");

      } // loop over kernel reps
      stopTimer();
    }

    if constexpr (tune_idx == 1) {

      constexpr bool async = true;

      using launch_policy = RAJA::LaunchPolicy<
      RAJA::hip_launch_t<async, mvpa::Q1D * mvpa::Q1D * mvpa::Q1D>>;

      using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

      using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;

      using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_y_direct>;

      using inner_z = RAJA::LoopPolicy<RAJA::hip_thread_z_direct>;

      using launch_context = RAJA::LaunchContext;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
        MASSVEC3DPA_HIP_RAJA_LAUNCH;
        RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");

      } // loop over kernel reps
      stopTimer();
    }

    if constexpr (tune_idx == 2) {

      constexpr bool async = true;

      using launch_policy = RAJA::LaunchPolicy<
      RAJA::hip_launch_t<async, mvpa::Q1D * mvpa::Q1D * mvpa::Q1D>>;

      using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

      using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_x_loop>;

      using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_y_loop>;

      using inner_z = RAJA::LoopPolicy<RAJA::hip_thread_z_loop>;

      //threadIdx, blockDim, blockIdx, gridDim cached
      using CachePolicy = RAJA::HipIndicesAndDims<false, false, true, false>;
      using launch_context =
          RAJA::LaunchContextT<
              RAJA::HipLaunchContextIndicesAndDimsPolicy<CachePolicy>>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
        MASSVEC3DPA_HIP_RAJA_LAUNCH;
        RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");

      } // loop over kernel reps
      stopTimer();
    }

    break;
  }

  default: {

    getCout() << "\n MASSVEC3DPA : Unknown Hip variant id = " << vid
              << std::endl;
    break;
  }
  }
}

#undef MASSVEC3DPA_HIP_RAJA_LAUNCH
#undef MASSVEC3DPA_GPU_KERNEL_BODY


void MASSVEC3DPA::defineHipVariantTunings()
{

  for (VariantID vid : {Base_HIP, RAJA_HIP}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (vid == Base_HIP) {

          addVariantTuning<&MASSVEC3DPA::runHipVariantImpl<block_size, 0>>(
              vid, "runtime_block_stride_loop_" + std::to_string(block_size));

          addVariantTuning<&MASSVEC3DPA::runHipVariantImpl<block_size, 1>>(
              vid, "direct_" + std::to_string(block_size));

        }

        if (vid == RAJA_HIP) {

          addVariantTuning<&MASSVEC3DPA::runHipVariantImpl<block_size, 0>>(
              vid, "runtime_block_stride_loop_" + std::to_string(block_size));

          addVariantTuning<&MASSVEC3DPA::runHipVariantImpl<block_size, 1>>(
              vid, "direct_" + std::to_string(block_size));

          addVariantTuning<&MASSVEC3DPA::runHipVariantImpl<block_size, 2>>(
              vid, "cached_block_stride_loop_" + std::to_string(block_size));

        }

      }

    });

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
