//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"
#include "common/MemPool.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_SETUP_OMP_TARGET \
  TRIAD_PARTED_FUSED_MANUAL_FUSER_SETUP \
  triad_holder* triad_holders; \
  allocData(DataSpace::OmpTarget, triad_holders, (parts.size()-1));

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_COPY_OMP_TARGET \
  initOpenMPDeviceData(omp_triad_holders, triad_holders, index*sizeof(triad_holder));

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN_OMP_TARGET \
  deallocData(DataSpace::OmpTarget, triad_holders); \
  TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN

void TRIAD_PARTED_FUSED::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    TRIAD_PARTED_FUSED_MANUAL_FUSER_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type index = 0;
      Index_type len_sum = 0;

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        triad_holders[index] = triad_holder{iend-ibegin, a, b, c, alpha, ibegin};
        len_sum += iend-ibegin;
        index += 1;
      }

      TRIAD_PARTED_FUSED_MANUAL_FUSER_COPY_OMP_TARGET;
      Index_type len_ave = (len_sum + index-1) / index;
      #pragma omp target is_device_ptr(a_ptrs, b_ptrs, c_ptrs, alpha_ptrs, ibegin_ptrs, len_ptrs) device( did )
      #pragma omp teams distribute parallel for collapse(2) schedule(static, 1)
      for (Index_type j = 0; j < index; j++) {
        for (Index_type iii = 0; iii < len_ave; iii++) {

          Index_type len    = omp_triad_holders[j].len;
          Real_ptr   a      = omp_triad_holders[j].a;
          Real_ptr   b      = omp_triad_holders[j].b;
          Real_ptr   c      = omp_triad_holders[j].c;
          Real_type  alpha  = omp_triad_holders[j].alpha;
          Index_type ibegin = omp_triad_holders[j].ibegin;

          for (Index_type ii = iii; ii < len; ii += len_ave) {
            Index_type i = ii + ibegin;
            TRIAD_PARTED_FUSED_BODY;
          }
        }
      }

    }
    stopTimer();

    TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    using AllocatorHolder = RAJAPoolAllocatorHolder<
        rajaperf::basic_mempool::MemPool<dataspace_allocator<getFuserDataSpace(RAJA_OpenMPTarget)>>>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::omp_target_work /*<threads_per_team>*/,
                                 RAJA::ordered,
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

        auto triad_parted_fused_lam = [=](Index_type i) {
              TRIAD_PARTED_FUSED_BODY;
            };

        pool.enqueue(
            RAJA::TypedRangeSegment<Index_type>(ibegin, iend),
            triad_parted_fused_lam );
      }
      workgroup group = pool.instantiate();
      worksite site = group.run();

    }
    stopTimer();

  } else {
     getCout() << "\n  TRIAD_PARTED_FUSED : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

