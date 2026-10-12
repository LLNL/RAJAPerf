//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <rocprim/block/block_reduce.hpp>
#include <rocprim/warp/warp_reduce.hpp>

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

const size_t warp_size = RAJA_HIP_WAVESIZE;

template < size_t block_size, size_t replication >
__launch_bounds__(block_size)
__global__ void atomic_replicate_thread(Real_ptr atomic,
                          Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_HIP, i, ATOMIC_VALUE);
  }
}

template < size_t block_size, size_t replication >
__launch_bounds__(block_size)
__global__ void atomic_replicate_warp(Real_ptr atomic,
                          Index_type iend)
{
  Real_type val = 0;

  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    val = ATOMIC_VALUE;
  }

  using WarpReduce = rocprim::warp_reduce<Real_type, warp_size>;
  __shared__ typename WarpReduce::storage_type warp_reduce_storage;
  WarpReduce().reduce(val, val, warp_reduce_storage);
  if ((threadIdx.x % warp_size) == 0) {
    ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_HIP, i/warp_size, val);
  }
}

template < size_t block_size, size_t replication >
__launch_bounds__(block_size)
__global__ void atomic_replicate_block(Real_ptr atomic,
                          Index_type iend)
{
  Real_type val = 0;

  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    val = ATOMIC_VALUE;
  }

  using BlockReduce = rocprim::block_reduce<Real_type, block_size>;
  __shared__ typename BlockReduce::storage_type block_reduce_storage;
  BlockReduce().reduce(val, val, block_reduce_storage);
  if (threadIdx.x == 0) {
    ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_HIP, blockIdx.x, val);
  }
}


template < size_t block_size, size_t replication >
void ATOMIC::runHipVariantReplicateGlobal(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  ATOMIC_DATA_SETUP(replication);

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (atomic_replicate_thread<block_size, replication>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend );
      RP_CALI_SUBKERNEL_END("ATOMIC_1");

    }
    stopTimer();

  } else  if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
      RAJA::forall<RAJA::hip_exec<block_size, true /*async*/>>( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_RAJA_HIP, i, ATOMIC_VALUE);
      });
      RP_CALI_SUBKERNEL_END("ATOMIC_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(replication);
}

template < size_t block_size, size_t replication >
void ATOMIC::runHipVariantReplicateWarp(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  ATOMIC_DATA_SETUP(replication);

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (atomic_replicate_warp<block_size, replication>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend );
      RP_CALI_SUBKERNEL_END("ATOMIC_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(replication);
}

template < size_t block_size, size_t replication >
void ATOMIC::runHipVariantReplicateBlock(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  ATOMIC_DATA_SETUP(replication);

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (atomic_replicate_block<block_size, replication>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend );
      RP_CALI_SUBKERNEL_END("ATOMIC_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(replication);
}


void ATOMIC::defineHipVariantTunings()
{

  for (VariantID vid : {Base_HIP, RAJA_HIP}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(replication)) {

            addVariantTuning<&ATOMIC::runHipVariantReplicateGlobal<block_size, replication>>(
                vid, "replicate_"+std::to_string(replication)+
                     "_global_"+std::to_string(block_size));

          }

        });

        if ( vid == Base_HIP ) {

          seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

            if (run_params.numValidAtomicReplication() == 0u ||
                run_params.validAtomicReplication(replication)) {

              addVariantTuning<&ATOMIC::runHipVariantReplicateWarp<block_size, replication>>(
                  vid, "replicate_"+std::to_string(replication)+
                       "_warp_"+std::to_string(block_size));

            }

          });

          seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

            if (run_params.numValidAtomicReplication() == 0u ||
                run_params.validAtomicReplication(replication)) {

              addVariantTuning<&ATOMIC::runHipVariantReplicateBlock<block_size, replication>>(
                  vid, "replicate_"+std::to_string(replication)+
                       "_block_"+std::to_string(block_size));

            }

          });

        }

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
