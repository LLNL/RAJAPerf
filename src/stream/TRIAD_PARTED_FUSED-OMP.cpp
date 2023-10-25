//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{


void TRIAD_PARTED_FUSED::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_FUSED_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      TRIAD_PARTED_FUSED_MANUAL_FUSER_SETUP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type index = 0;

        for (size_t p = 1; p < parts.size(); ++p ) {
          const Index_type ibegin = parts[p-1];
          const Index_type iend = parts[p];

          triad_holders[index] = triad_holder{a, b, c, alpha, ibegin};
          lens[index]          = iend-ibegin;
          index += 1;
        }

#if defined(RAJA_ENABLE_OMP_TASK_INTERNAL)
        #pragma omp parallel
        #pragma omp single nowait
        for (Index_type j = 0; j < index; j++) {
          #pragma omp task firstprivate(j)
          {
            Real_ptr   a      = triad_holders[j].a;
            Real_ptr   b      = triad_holders[j].b;
            Real_ptr   c      = triad_holders[j].c;
            Real_type  alpha  = triad_holders[j].alpha;
            Index_type ibegin = triad_holders[j].ibegin;
            Index_type len    = lens[j];
            for (Index_type ii = 0; ii < len; ++ii ) {
              Index_type i = ii + ibegin;
              TRIAD_PARTED_FUSED_BODY;
            }
          }
        }
#else
        #pragma omp parallel for
        for (Index_type j = 0; j < index; j++) {
          Real_ptr   a      = triad_holders[j].a;
          Real_ptr   b      = triad_holders[j].b;
          Real_ptr   c      = triad_holders[j].c;
          Real_type  alpha  = triad_holders[j].alpha;
          Index_type ibegin = triad_holders[j].ibegin;
          Index_type len    = lens[j];
          for (Index_type ii = 0; ii < len; ++ii ) {
            Index_type i = ii + ibegin;
            TRIAD_PARTED_FUSED_BODY;
          }
        }
#endif

      }
      stopTimer();

      TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN;

      break;
    }

    case Lambda_OpenMP : {

      TRIAD_PARTED_FUSED_MANUAL_LAMBDA_FUSER_SETUP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type index = 0;

        for (size_t p = 1; p < parts.size(); ++p ) {
          const Index_type ibegin = parts[p-1];
          const Index_type iend = parts[p];

          new(&lambdas[index]) lambda_type(make_lambda(a, b, c, alpha, ibegin));
          lens[index] = iend-ibegin;
          index += 1;
        }

#if defined(RAJA_ENABLE_OMP_TASK_INTERNAL)
        #pragma omp parallel
        #pragma omp single nowait
        for (Index_type j = 0; j < index; j++) {
          #pragma omp task firstprivate(j)
          {
            auto       lambda = lambdas[j];
            Index_type len    = lens[j];
            for (Index_type ii = 0; ii < len; ii++) {
              lambda(ii);
            }
          }
        }
#else
        #pragma omp parallel for
        for (Index_type j = 0; j < index; j++) {
          auto       lambda = lambdas[j];
          Index_type len    = lens[j];
          for (Index_type ii = 0; ii < len; ii++) {
            lambda(ii);
          }
        }
#endif


      }
      stopTimer();

      TRIAD_PARTED_FUSED_MANUAL_LAMBDA_FUSER_TEARDOWN;

      break;
    }

    case RAJA_OpenMP : {

      using AllocatorHolder = RAJAPoolAllocatorHolder<
        RAJA::basic_mempool::MemPool<RAJA::basic_mempool::generic_allocator>>;
      using Allocator = AllocatorHolder::Allocator<char>;

      AllocatorHolder allocatorHolder;

      using workgroup_policy = RAJA::WorkGroupPolicy <
                                   RAJA::omp_work,
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

      workpool pool  (allocatorHolder.template getAllocator<char>());
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

      break;
    }

    default : {
      getCout() << "\n  TRIAD_PARTED_FUSED : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace stream
} // end namespace rajaperf
