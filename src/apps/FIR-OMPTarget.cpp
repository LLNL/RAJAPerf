//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define FIR_DATA_SETUP_OMP_TARGET \
  Real_ptr coeff; \
  \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocData(DataSpace::OmpTarget, coeff, FIR_COEFFLEN); \
  copyData(DataSpace::OmpTarget, coeff, DataSpace::Host, tcoeff, FIR_COEFFLEN);


#define FIR_DATA_TEARDOWN_OMP_TARGET \
  deallocData(DataSpace::OmpTarget, coeff);


void FIR::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize() - m_coefflen;

  FIR_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(in, out, coeff) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
         FIR_BODY;
      }

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        FIR_BODY;
      });

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else {
     getCout() << "\n  FIR : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
