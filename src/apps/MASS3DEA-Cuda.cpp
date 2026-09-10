//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DEA(const Real_ptr B, const Real_ptr D, Real_ptr M) {

  const Index_type e = blockIdx.x;

  MASS3DEA_0

  GPU_FOREACH_THREAD_INC(iz, z, 1, mea::D1D) {
    GPU_FOREACH_THREAD_INC(d, x, mea::D1D, mea::D1D) {
      GPU_FOREACH_THREAD_INC(q, y, mea::Q1D, mea::D1D) {
        MASS3DEA_1
      }
    }
  }

  MASS3DEA_2

  GPU_FOREACH_THREAD_INC(k1, x, mea::Q1D, mea::D1D) {
    GPU_FOREACH_THREAD_INC(k2, y, mea::Q1D, mea::D1D) {
      GPU_FOREACH_THREAD_INC(k3, z, mea::Q1D, mea::D1D) {
        MASS3DEA_3
      }
    }
  }

  __syncthreads();

  GPU_FOREACH_THREAD_INC(i1, x, mea::D1D, mea::D1D) {
    GPU_FOREACH_THREAD_INC(i2, y, mea::D1D, mea::D1D) {
      GPU_FOREACH_THREAD_INC(i3, z, mea::D1D, mea::D1D) {
        MASS3DEA_4
      }
    }
  }

}

#define MASS3DEA_CUDA_RAJA_LAUNCH                                             \
  {                                                                           \
                                                                              \
    /* clang-format off */                                                    \
    RAJA::launch<launch_policy>(                                              \
      res,                                                                    \
      RAJA::LaunchParams(RAJA::Teams(NE),                                     \
                         RAJA::Threads(mea::D1D, mea::D1D, mea::D1D)),        \
      [=] RAJA_HOST_DEVICE(launch_context ctx) {                              \
                                                                              \
        RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),                   \
          [&](Index_type e) {                                                 \
                                                                              \
            MASS3DEA_0                                                        \
                                                                              \
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),                \
              [&](Index_type) {                                               \
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),     \
                  [&](Index_type d) {                                         \
                    RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D), \
                      [&](Index_type q) {                                     \
                        MASS3DEA_1                                            \
                      }                                                       \
                    );                                                        \
                  }                                                           \
                );                                                            \
              }                                                               \
            );                                                                \
                                                                              \
            MASS3DEA_2                                                        \
                                                                              \
            RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::Q1D),         \
              [&](Index_type k1) {                                            \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),     \
                  [&](Index_type k2) {                                        \
                    RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::Q1D), \
                      [&](Index_type k3) {                                    \
                        MASS3DEA_3                                            \
                      }                                                       \
                    );                                                        \
                  }                                                           \
                );                                                            \
              }                                                               \
            );                                                                \
                                                                              \
            ctx.teamSync();                                                   \
                                                                              \
            RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),         \
              [&](Index_type i1) {                                            \
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::D1D),     \
                  [&](Index_type i2) {                                        \
                    RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::D1D), \
                      [&](Index_type i3) {                                    \
                        MASS3DEA_4                                            \
                      }                                                       \
                    );                                                        \
                  }                                                           \
                );                                                            \
              }                                                               \
            );                                                                \
                                                                              \
          }                                                                   \
        );                                                                    \
                                                                              \
      }                                                                       \
    );                                                                        \
    /* clang-format on */                                                     \
                                                                              \
  }

template <size_t block_size, size_t tune_idx>
void MASS3DEA::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_CUDA: {

    if constexpr (tune_idx == 0) {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
        dim3 nthreads_per_block(mea::D1D, mea::D1D, mea::D1D);
        constexpr size_t shmem = 0;

        RPlaunchCudaKernel((Mass3DEA<block_size>), NE, nthreads_per_block, shmem,
                           res.get_stream(), B, D, M);
        RP_CALI_SUBKERNEL_END("MASS3DEA_1");
      }
      stopTimer();
    }
    break;
  }

  case RAJA_CUDA: {

    if constexpr (tune_idx == 0) {

      constexpr bool async = true;

      using launch_policy = RAJA::LaunchPolicy<
          RAJA::cuda_launch_t<async, mea::D1D * mea::D1D * mea::D1D>>;

      using outer_x = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

      using inner_x = RAJA::LoopPolicy<RAJA::cuda_thread_size_x_loop<mea::D1D>>;

      using inner_y = RAJA::LoopPolicy<RAJA::cuda_thread_size_y_loop<mea::D1D>>;

      using inner_z = RAJA::LoopPolicy<RAJA::cuda_thread_size_z_loop<mea::D1D>>;

      using launch_context = RAJA::LaunchContext;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
        MASS3DEA_CUDA_RAJA_LAUNCH;
        RP_CALI_SUBKERNEL_END("MASS3DEA_1");

      }  // loop over kernel reps
      stopTimer();

    } else if constexpr (tune_idx == 1) {

      constexpr bool async = true;

      using launch_policy = RAJA::LaunchPolicy<
          RAJA::cuda_launch_t<async, mea::D1D * mea::D1D * mea::D1D>>;

      using outer_x = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

      using inner_x = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;

      using inner_y = RAJA::LoopPolicy<RAJA::cuda_thread_y_loop>;

      using inner_z = RAJA::LoopPolicy<RAJA::cuda_thread_z_loop>;

      // threadIdx, blockDim, blockIdx, gridDim cached
      using CachePolicy = RAJA::CudaIndicesAndDims<false, false, true, false>;
      using launch_context =
          RAJA::LaunchContextT<
              RAJA::CudaLaunchContextIndicesAndDimsPolicy<CachePolicy>>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
        MASS3DEA_CUDA_RAJA_LAUNCH;
        RP_CALI_SUBKERNEL_END("MASS3DEA_1");

      }  // loop over kernel reps
      stopTimer();
    }

    break;
  }

  default: {

    getCout() << "\n MASS3DEA : Unknown Cuda variant id = " << vid << std::endl;
    break;
  }
  }
}

#undef MASS3DEA_CUDA_RAJA_LAUNCH

void MASS3DEA::defineCudaVariantTunings()
{

  for (VariantID vid : {Base_CUDA, RAJA_CUDA}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (vid == Base_CUDA) {

          addVariantTuning<&MASS3DEA::runCudaVariantImpl<block_size, 0>>(
              vid, "compile_time_block_stride_loop_" + std::to_string(block_size));

        }

        if (vid == RAJA_CUDA) {

          addVariantTuning<&MASS3DEA::runCudaVariantImpl<block_size, 0>>(
              vid, "compile_time_block_stride_loop_" + std::to_string(block_size));

          addVariantTuning<&MASS3DEA::runCudaVariantImpl<block_size, 1>>(
              vid, "cached_block_stride_loop_" + std::to_string(block_size));

        }

      }

    });

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
