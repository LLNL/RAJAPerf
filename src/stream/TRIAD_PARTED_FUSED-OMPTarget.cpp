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
  void** ptrs; \
  allocData(DataSpace::OmpTarget, ptrs, 6 * (parts.size()-1)); \
  Real_ptr*   a_ptrs      = reinterpret_cast<Real_ptr*>(ptrs) + 0 * (parts.size()-1); \
  Real_ptr*   b_ptrs      = reinterpret_cast<Real_ptr*>(ptrs) + 1 * (parts.size()-1); \
  Real_ptr*   c_ptrs      = reinterpret_cast<Real_ptr*>(ptrs) + 2 * (parts.size()-1); \
  Real_type*  alpha_ptrs  = reinterpret_cast<Real_type*>(ptrs) + 3 * (parts.size()-1); \
  Index_type* ibegin_ptrs = reinterpret_cast<Index_type*>(ptrs) + 4 * (parts.size()-1); \
  Index_type* len_ptrs    = reinterpret_cast<Index_type*>(ptrs) + 5 * (parts.size()-1); \
  void** h_ptrs = new void*[6 * (parts.size()-1)]; \
  Real_ptr*   h_a_ptrs      = reinterpret_cast<Real_ptr*>(h_ptrs) + 0 * (parts.size()-1); \
  Real_ptr*   h_b_ptrs      = reinterpret_cast<Real_ptr*>(h_ptrs) + 1 * (parts.size()-1); \
  Real_ptr*   h_c_ptrs      = reinterpret_cast<Real_ptr*>(h_ptrs) + 2 * (parts.size()-1); \
  Real_type*  h_alpha_ptrs  = reinterpret_cast<Real_type*>(h_ptrs) + 3 * (parts.size()-1); \
  Index_type* h_ibegin_ptrs = reinterpret_cast<Index_type*>(h_ptrs) + 4 * (parts.size()-1); \
  Index_type* h_len_ptrs    = reinterpret_cast<Index_type*>(h_ptrs) + 5 * (parts.size()-1);

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_COPY_OMP_TARGET \
  initOpenMPDeviceData(ptrs, h_ptrs, 4 * num_neighbors * num_vars);

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN_OMP_TARGET \
  deallocData(DataSpace::OmpTarget, ptrs); \
  delete[] h_ptrs;

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

        h_a_ptrs[index] = a;
        h_b_ptrs[index] = b;
        h_c_ptrs[index] = c;
        h_alpha_ptrs[index] = alpha;
        h_ibegin_ptrs[index] = ibegin;
        h_len_ptrs[index] = len;
        len_sum += len;
        index += 1;

        #pragma omp target is_device_ptr(a, b, c) device( did )
        #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIAD_PARTED_FUSED_BODY;
        }
      }
      TRIAD_PARTED_FUSED_MANUAL_FUSER_COPY_OMP_TARGET;
      Index_type len_ave = (len_sum + index-1) / index;
      #pragma omp target is_device_ptr(a_ptrs, b_ptrs, c_ptrs, alpha_ptrs, ibegin_ptrs, len_ptrs) device( did )
      #pragma omp teams distribute parallel for collapse(2) schedule(static, 1)
      for (Index_type j = 0; j < index; j++) {
        for (Index_type iii = 0; iii < len_ave; iii++) {

          Real_ptr   c      = a_ptrs[j];
          Real_ptr   b      = b_ptrs[j];
          Real_ptr   c      = c_ptrs[j];
          Real_type  alpha  = alpha_ptrs[j];
          Index_type ibegin = ibegin_ptrs[j];
          Index_type len    = len_ptrs[j];

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
        RAJA::basic_mempool::MemPool<RAJA::basic_mempool::generic_allocator>>;
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

