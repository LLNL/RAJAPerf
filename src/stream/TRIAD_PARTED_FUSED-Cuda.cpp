//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_SETUP_CUDA \
  Real_ptr*   a_ptrs; \
  Real_ptr*   b_ptrs; \
  Real_ptr*   c_ptrs; \
  Real_type*  alpha_ptrs; \
  Index_type* ibegin_ptrs; \
  Index_type* len_ptrs; \
  allocData(DataSpace::CudaPinned, a_ptrs, parts.size()-1); \
  allocData(DataSpace::CudaPinned, b_ptrs, parts.size()-1); \
  allocData(DataSpace::CudaPinned, c_ptrs, parts.size()-1); \
  allocData(DataSpace::CudaPinned, alpha_ptrs, parts.size()-1); \
  allocData(DataSpace::CudaPinned, ibegin_ptrs, parts.size()-1); \
  allocData(DataSpace::CudaPinned, len_ptrs, parts.size()-1);

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN_CUDA \
  deallocData(DataSpace::CudaPinned, a_ptrs); \
  deallocData(DataSpace::CudaPinned, b_ptrs); \
  deallocData(DataSpace::CudaPinned, c_ptrs); \
  deallocData(DataSpace::CudaPinned, alpha_ptrs); \
  deallocData(DataSpace::CudaPinned, ibegin_ptrs); \
  deallocData(DataSpace::CudaPinned, len_ptrs);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void triad_parted_fused(Real_ptr* a_ptrs, Real_ptr* b_ptrs,
                                   Real_ptr* c_ptrs, Real_type* alpha_ptrs,
                                   Index_type* ibegin_ptrs, Index_type* len_ptrs)
{
  Index_type j = blockIdx.y;

  Real_ptr   a = a_ptrs[j];
  Real_ptr   b = b_ptrs[j];
  Real_ptr   c = c_ptrs[j];
  Real_type  alpha = alpha_ptrs[j];
  Index_type ibegin = ibegin_ptrs[j];
  Index_type iend = ibegin + len_ptrs[j];

  for (Index_type i = ibegin + threadIdx.x + blockIdx.x * block_size;
       i < iend;
       i += block_size * gridDim.x) {
    TRIAD_PARTED_FUSED_BODY;
  }
}


template < size_t block_size >
void TRIAD_PARTED_FUSED::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    TRIAD_PARTED_FUSED_MANUAL_FUSER_SETUP_CUDA

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type index = 0;
      Index_type len_sum = 0;

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        a_ptrs[index] = a;
        b_ptrs[index] = b;
        c_ptrs[index] = c;
        alpha_ptrs[index] = alpha;
        ibegin_ptrs[index] = ibegin;
        len_ptrs[index] = iend-ibegin;
        len_sum += iend-ibegin;
        index += 1;
      }
      Index_type len_ave = (len_sum + index-1) / index;
      dim3 nthreads_per_block(block_size);
      dim3 nblocks((len_ave + block_size-1) / block_size, index);
      constexpr size_t shmem = 0;
      triad_parted_fused<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(
          a_ptrs, b_ptrs, c_ptrs, alpha_ptrs, ibegin_ptrs, len_ptrs);
      cudaErrchk( cudaGetLastError() );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );

    }
    stopTimer();

    TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN_CUDA

  } else if ( vid == RAJA_CUDA ) {

    using AllocatorHolder = RAJAPoolAllocatorHolder<RAJA::cuda::pinned_mempool_type>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::cuda_work_async<block_size>,
                                 RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    workpool pool(allocatorHolder.template getAllocator<char>());
    pool.reserve(parts.size()-1, 1024ull*1024ull);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        auto triad_parted_fused_lam = [=] __device__ (Index_type i) {
              TRIAD_PARTED_FUSED_BODY;
            };

        pool.enqueue(
            RAJA::TypedRangeSegment<Index_type>(ibegin, iend),
            triad_parted_fused_lam );
      }
      workgroup group = pool.instantiate();
      worksite site = group.run(res);
      res.wait();

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED_FUSED : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRIAD_PARTED_FUSED, Cuda)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
