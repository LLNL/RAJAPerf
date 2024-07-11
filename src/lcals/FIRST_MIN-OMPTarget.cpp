//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void FIRST_MIN::runOpenMPTargetVariant(VariantID vid, size_t tune_idx)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_MIN_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp declare reduction(minloc : MyMinLoc : \
                                    omp_out = MinLoc_compare(omp_out, omp_in))\
                                    initializer (omp_priv = omp_orig)

      FIRST_MIN_MINLOC_INIT;

      #pragma omp target is_device_ptr(x) device( did ) map(tofrom:mymin)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) \
                  reduction(minloc:mymin)
      for (Index_type i = ibegin; i < iend; ++i ) {
        FIRST_MIN_BODY;
      }

      m_minloc = mymin.loc;

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    if (tune_idx == 0) {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceMinLoc<RAJA::omp_target_reduce, Real_type, Index_type> loc(
                                                    m_xmin_init, m_initloc);

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          FIRST_MIN_BODY_RAJA;
        });

        m_minloc = loc.getLoc();

      }
      stopTimer();

    } else if (tune_idx == 1) {

      using VL_TYPE = RAJA::expt::ValLoc<Real_type>;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        VL_TYPE tloc(m_xmin_init, m_initloc);

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend),
          RAJA::expt::Reduce<RAJA::operators::minimum>(&tloc),
          [=](Index_type i, VL_TYPE& loc) {
            loc.min(x[i], i);
          }
        );

        m_minloc = static_cast<Index_type>(tloc.getLoc());

      }
      stopTimer();

    } else {
      getCout() << "\n  FIRST_MIN : Unknown OMP Target tuning index = " << tune_idx << std::endl;
    }

  } else {
     getCout() << "\n  FIRST_MIN : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

void FIRST_MIN::setOpenMPTargetTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_OpenMPTarget) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
