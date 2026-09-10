//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"
#include "common/CudaGridScan.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
using cuda_items_per_thread_type = integer::make_gpu_items_per_thread_list_type<
    detail::cuda::grid_scan_max_items_per_thread<Index_type, block_size>::value+1,
    integer::LessEqual<detail::cuda::grid_scan_max_items_per_thread<Index_type, block_size>::value>>;


template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void indexlist_custom(Real_ptr x,
                                 Int_ptr list,
                                 Index_ptr block_counts,
                                 Index_ptr grid_counts,
                                 unsigned* block_readys,
                                 Index_ptr len,
                                 Index_type iend)
{
  // blocks do start running in order in cuda, so a block with a higher
  // index can wait on a block with a lower index without deadlocking
  // (replace with an atomicInc if this changes)
  const int block_id = blockIdx.x;

  Index_type vals[items_per_thread];

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    Index_type val = 0;
    if (i < iend) {
      if (INDEXLIST_CONDITIONAL) {
        val = 1;
      }
    }
    vals[ti] = val;
  }

  Index_type exclusives[items_per_thread];
  Index_type inclusives[items_per_thread];
  detail::cuda::GridScan<Index_type, block_size, items_per_thread>::grid_scan(
      block_id, vals, exclusives, inclusives, block_counts, grid_counts, block_readys);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    Index_type exclusive = exclusives[ti];
    Index_type inclusive = inclusives[ti];
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
void INDEXLIST::runCudaVariantCustom(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  INDEXLIST_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*items_per_thread);
    const size_t shmem_size = 0;

    Index_ptr len;
    allocData(DataSpace::CudaPinned, len, 1);
    Index_ptr block_counts;
    allocData(DataSpace::CudaDevice, block_counts, grid_size);
    Index_ptr grid_counts;
    allocData(DataSpace::CudaDevice, grid_counts, grid_size);
    unsigned* block_readys;
    allocData(DataSpace::CudaDevice, block_readys, grid_size);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_1");
      CAMP_CUDA_API_INVOKE_AND_CHECK( cudaMemsetAsync,
          block_readys, 0, sizeof(unsigned)*grid_size, res.get_stream() );
      RPlaunchCudaKernel( (indexlist_custom<block_size, items_per_thread>),
                          grid_size, block_size,
                          shmem_size, res.get_stream(),
                          x+ibegin, list+ibegin,
                          block_counts, grid_counts, block_readys,
                          len, iend-ibegin );

      CAMP_CUDA_API_INVOKE_AND_CHECK( cudaStreamSynchronize, res.get_stream() );
      m_len = *len;
      RP_CALI_SUBKERNEL_END("INDEXLIST_1");

    }
    stopTimer();

    deallocData(DataSpace::CudaPinned, len);
    deallocData(DataSpace::CudaDevice, block_counts);
    deallocData(DataSpace::CudaDevice, grid_counts);
    deallocData(DataSpace::CudaDevice, block_readys);

  } else {
    getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
  }
}


void INDEXLIST::defineCudaVariantTunings()
{

  for (VariantID vid : {Base_CUDA}) {

    if ( vid == Base_CUDA && run_params.getEnableCustomScan() ) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          using cuda_items_per_thread = cuda_items_per_thread_type<block_size>;

          if (camp::size<cuda_items_per_thread>::value == 0) {

            addVariantTuning<&INDEXLIST::runCudaVariantCustom<
                                 decltype(block_size)::value,
                                 detail::cuda::grid_scan_default_items_per_thread<
                                     Real_type, block_size,
                                     RAJA_PERFSUITE_TUNING_CUDA_ARCH>::value>>(
                vid, "block_"+std::to_string(block_size));

          }

          seq_for(cuda_items_per_thread{}, [&](auto items_per_thread) {

            if (run_params.numValidItemsPerThread() == 0u ||
                run_params.validItemsPerThread(block_size)) {

              addVariantTuning<&INDEXLIST::runCudaVariantCustom<
                                   decltype(block_size)::value,
                                   items_per_thread>>(
                  vid, "itemsPerThread<"+std::to_string(items_per_thread)+">_"
                       "block_"+std::to_string(block_size));

            }

          });

        }

      });

    }

  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
