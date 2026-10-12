//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void DAXPY_ATOMIC::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DAXPY_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DAXPY_ATOMIC_1");
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          DAXPY_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_OMP);
        }
        RP_CALI_SUBKERNEL_END("DAXPY_ATOMIC_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto daxpy_atomic_lam = [=](Index_type i) {
            DAXPY_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_OMP);
          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DAXPY_ATOMIC_1");
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          daxpy_atomic_lam(i);
        }
        RP_CALI_SUBKERNEL_END("DAXPY_ATOMIC_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DAXPY_ATOMIC_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DAXPY_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_RAJA_OMP);
        });
        RP_CALI_SUBKERNEL_END("DAXPY_ATOMIC_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  DAXPY_ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DAXPY_ATOMIC, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace basic
} // end namespace rajaperf
