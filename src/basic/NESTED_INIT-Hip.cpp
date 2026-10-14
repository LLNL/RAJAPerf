//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block shape for Hip execution
  //
#define i_block_sz (32)
#define j_block_sz (block_size / i_block_sz)
#define k_block_sz (1)

#define NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  i_block_sz, j_block_sz, k_block_sz

#define NESTED_INIT_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP); \
  static_assert(i_block_sz*j_block_sz*k_block_sz == block_size, "Invalid block_size");

#define NESTED_INIT_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, i_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nk, k_block_sz)));


template< size_t i_block_size, size_t j_block_size, size_t k_block_size >
  __launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init(Real_ptr array,
                            Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.x * i_block_size + threadIdx.x;
  Index_type j = blockIdx.y * j_block_size + threadIdx.y;
  Index_type k = blockIdx.z;

  if ( i < ni && j < nj && k < nk ) {
    NESTED_INIT_BODY;
  }
}

template< size_t i_block_size, size_t j_block_size, size_t k_block_size, typename Lambda >
__launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init_lam(Index_type ni, Index_type nj, Index_type nk,
                                Lambda body)
{
  Index_type i = blockIdx.x * i_block_size + threadIdx.x;
  Index_type j = blockIdx.y * j_block_size + threadIdx.y;
  Index_type k = blockIdx.z;

  if ( i < ni && j < nj && k < nk ) {
    body(i, j, k);
  }
}



template < size_t block_size >
void NESTED_INIT::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      NESTED_INIT_THREADS_PER_BLOCK_HIP;
      NESTED_INIT_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      RPlaunchHipKernel(
        (nested_init<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        array, ni, nj, nk );
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      auto nested_init_lambda = [=] __device__ (Index_type i, 
                                                Index_type j,
                                                Index_type k) {
        NESTED_INIT_BODY;
      };

      NESTED_INIT_THREADS_PER_BLOCK_HIP;
      NESTED_INIT_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      RPlaunchHipKernel(
        (nested_init_lam<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                         decltype(nested_init_lambda)>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        ni, nj, nk,
        nested_init_lambda );
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    using EXEC_POL =
      RAJA::KernelPolicy<
       RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<2, RAJA::hip_block_z_direct,      // k
            RAJA::statement::For<1, RAJA::hip_global_size_y_direct<j_block_sz>,   // j
              RAJA::statement::For<0, RAJA::hip_global_size_x_direct<i_block_sz>, // i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;


    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                         RAJA::RangeSegment(0, nj),
                         RAJA::RangeSegment(0, nk)),
        res,
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void NESTED_INIT::runHipVariantFornest(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    using EXEC_POL =
      RAJA::fornest_collapsed_policy<RAJA::hip_exec<block_size, true /*async*/>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    [=] __device__ (Index_type k, Index_type j, Index_type i) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size, size_t tile_k, size_t tile_j, size_t tile_i >
void NESTED_INIT::runHipVariantFornestRuntimeTiled(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    using EXEC_POL =
      RAJA::fornest_tiling_policy<RAJA::hip_exec<block_size, true /*async*/>,
                                  RAJA::fornest_tile_runtime,
                                  RAJA::fornest_tile_runtime,
                                  RAJA::fornest_tile_runtime>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(res,
                    EXEC_POL {RAJA::TileSize(tile_k),
                              RAJA::TileSize(tile_j),
                              RAJA::TileSize(tile_i)},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    [=] __device__ (Index_type k, Index_type j, Index_type i) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void NESTED_INIT::runHipVariantFornestAutoTiled(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    using EXEC_POL =
      RAJA::fornest_tiling_policy<RAJA::hip_exec<block_size, true /*async*/>,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    [=] __device__ (Index_type k, Index_type j, Index_type i) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Hip variant id = " << vid << std::endl;
  }
}

void NESTED_INIT::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuning<&NESTED_INIT::runHipVariantImpl<block_size>>(
            vid, "block_"+std::to_string(block_size));

        if (vid == RAJA_HIP) {
          addVariantTuning<&NESTED_INIT::runHipVariantFornest<block_size>>(
              vid, "fornest-collapse_block_"+std::to_string(block_size));
          addVariantTuning<&NESTED_INIT::runHipVariantFornestAutoTiled<block_size>>(
              vid, "fornest-auto-tile_block_"+std::to_string(block_size));

          if constexpr (decltype(block_size)::value == 1 * 8 * 32) {
            addVariantTuning<&NESTED_INIT::runHipVariantFornestRuntimeTiled<block_size, 1, 8, 32>>(
                vid, "fornest-runtime-tile_1x8x32_block_"+std::to_string(block_size));
          }
          if constexpr (decltype(block_size)::value == 2 * 4 * 32) {
            addVariantTuning<&NESTED_INIT::runHipVariantFornestRuntimeTiled<block_size, 2, 4, 32>>(
                vid, "fornest-runtime-tile_2x4x32_block_"+std::to_string(block_size));
          }
          if constexpr (decltype(block_size)::value == 4 * 4 * 16) {
            addVariantTuning<&NESTED_INIT::runHipVariantFornestRuntimeTiled<block_size, 4, 4, 16>>(
                vid, "fornest-runtime-tile_4x4x16_block_"+std::to_string(block_size));
          }
        }

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
