//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULTI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void MULTI_REDUCE::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MULTI_REDUCE_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      MULTI_REDUCE_SETUP_VALUES;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        MULTI_REDUCE_INIT_VALUES;

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          #pragma omp atomic
          MULTI_REDUCE_BODY;
        }

        MULTI_REDUCE_FINALIZE_VALUES;

      }
      stopTimer();

      MULTI_REDUCE_TEARDOWN_VALUES;

      break;
    }

    case Lambda_OpenMP : {

      MULTI_REDUCE_SETUP_VALUES;

      auto multi_reduce_base_lam = [=](Index_type i) {
                                 #pragma omp atomic
                                 MULTI_REDUCE_BODY;
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        MULTI_REDUCE_INIT_VALUES;

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          multi_reduce_base_lam(i);
        }

        MULTI_REDUCE_FINALIZE_VALUES;

      }
      stopTimer();

      MULTI_REDUCE_TEARDOWN_VALUES;

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        MULTI_REDUCE_INIT_VALUES_RAJA(RAJA::omp_multi_reduce);

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            MULTI_REDUCE_BODY;
        });

        MULTI_REDUCE_FINALIZE_VALUES_RAJA(RAJA::omp_multi_reduce);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  MULTI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

  MULTI_REDUCE_DATA_TEARDOWN;

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
