//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"
#include "common/HipGridScan.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
using hip_items_per_thread_type = integer::make_gpu_items_per_thread_list_type<
    detail::hip::grid_scan_default_items_per_thread<INDEXLIST::Idx_type, block_size, RAJA_PERFSUITE_TUNING_HIP_ARCH>::value,
    integer::LessEqual<detail::hip::grid_scan_max_items_per_thread<INDEXLIST::Idx_type, block_size>::value>>;


template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void indexlist(Real_ptr x,
                          INDEXLIST::Idx_ptr list,
                          INDEXLIST::Idx_type* block_counts,
                          INDEXLIST::Idx_type* grid_counts,
                          unsigned* block_readys,
                          INDEXLIST::Idx_type* len,
                          Index_type iend)
{
  // It looks like blocks do not start running in order in hip, so a block
  // with a higher index can't wait on a block with a lower index without
  // deadlocking (have to replace with an atomicInc)
  const int block_id = blockIdx.x;

  INDEXLIST::Idx_type vals[items_per_thread];

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    INDEXLIST::Idx_type val = 0;
    if (i < iend) {
      if (INDEXLIST_CONDITIONAL) {
        val = 1;
      }
    }
    vals[ti] = val;
  }

  INDEXLIST::Idx_type exclusives[items_per_thread];
  INDEXLIST::Idx_type inclusives[items_per_thread];
  detail::hip::GridScan<INDEXLIST::Idx_type, block_size, items_per_thread>::grid_scan(
      block_id, vals, exclusives, inclusives, block_counts, grid_counts, block_readys);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    INDEXLIST::Idx_type exclusive = exclusives[ti];
    INDEXLIST::Idx_type inclusive = inclusives[ti];
    if (i < iend) {
      if (exclusive != inclusive) {
        list[exclusive] = i;
      }
      if (i == iend-1) {
        *len = inclusive;
      }
    }
  }
}

template < size_t block_size, size_t items_per_thread >
void INDEXLIST::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  INDEXLIST_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*items_per_thread);
    const size_t shmem_size = 0;

    Idx_type* len;
    allocData(DataSpace::HipPinnedCoarse, len, 1);
    Idx_type* block_counts;
    allocData(DataSpace::HipDevice, block_counts, grid_size);
    Idx_type* grid_counts;
    allocData(DataSpace::HipDevice, grid_counts, grid_size);
    unsigned* block_readys;
    allocData(DataSpace::HipDevice, block_readys, grid_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipMemsetAsync(block_readys, 0, sizeof(unsigned)*grid_size,
                                res.get_stream()) );

      RPlaunchHipKernel( (indexlist<block_size, items_per_thread>),
                         grid_size, block_size,
                         shmem_size, res.get_stream(),
                         x+ibegin, list+ibegin,
                         block_counts, grid_counts, block_readys,
                         len, iend-ibegin );

      hipErrchk( hipStreamSynchronize( res.get_stream() ) );
      m_len = *len;

    }
    stopTimer();

    deallocData(DataSpace::HipPinnedCoarse, len);
    deallocData(DataSpace::HipDevice, block_counts);
    deallocData(DataSpace::HipDevice, grid_counts);
    deallocData(DataSpace::HipDevice, block_readys);

  } else {
    getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
  }
}


void INDEXLIST::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(hip_items_per_thread_type<block_size>{}, [&](auto items_per_thread) {

          if (run_params.numValidItemsPerThread() == 0u ||
              run_params.validItemsPerThread(block_size)) {

            if (tune_idx == t) {

              runHipVariantImpl<block_size, items_per_thread>(vid);

            }

            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  INDEXLIST : Unknown Hip variant id = " << vid << std::endl;

  }
}

void INDEXLIST::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(hip_items_per_thread_type<block_size>{}, [&](auto items_per_thread) {

          if (run_params.numValidItemsPerThread() == 0u ||
              run_params.validItemsPerThread(block_size)) {

            addVariantTuningName(vid, "block_"+std::to_string(block_size)+
                                      "_itemsPerThread_"+std::to_string(items_per_thread));

          }

        });

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
