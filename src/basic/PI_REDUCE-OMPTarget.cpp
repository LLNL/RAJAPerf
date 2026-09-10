//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void PI_REDUCE::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_REDUCE_1");
      Real_type pi = m_pi_init;

      #pragma omp target device( did ) map(tofrom:pi)
      #pragma omp teams distribute parallel for reduction(+:pi) \
              thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        PI_REDUCE_BODY;
      }

      m_pi = 4.0 * pi;
      RP_CALI_SUBKERNEL_END("PI_REDUCE_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_REDUCE_1");
      Real_type tpi = m_pi_init;

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tpi),
        [=] (Index_type i,
          RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& pi) {
          PI_REDUCE_BODY;
        }
      );

      m_pi = static_cast<Real_type>(tpi) * 4.0;
      RP_CALI_SUBKERNEL_END("PI_REDUCE_1");

    }
    stopTimer();

  } else {
    getCout() << "\n  PI_REDUCE : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PI_REDUCE, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
