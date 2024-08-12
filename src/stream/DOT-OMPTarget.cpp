//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

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

void DOT::runOpenMPTargetVariant(VariantID vid, size_t tune_idx)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMPTarget : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;

        #pragma omp target is_device_ptr(a, b) device( did ) map(tofrom:dot)
        #pragma omp teams distribute parallel for reduction(+:dot) \
                thread_limit(threads_per_team) schedule(static, 1)
        for (Index_type i = ibegin; i < iend; ++i ) {
          DOT_BODY;
        }

        m_dot += dot;

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMPTarget : {

      if (tune_idx == 0) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          RAJA::ReduceSum<RAJA::omp_target_reduce, Real_type> dot(m_dot_init);

          RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            DOT_BODY;
          });

          m_dot += static_cast<Real_type>(dot.get());

        }
        stopTimer();

      } else if (tune_idx == 1) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          Real_type tdot = m_dot_init;

          RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tdot),
            [=] (Index_type i, Real_type& dot) {
              DOT_BODY;
            }
          );

          m_dot += static_cast<Real_type>(tdot);

        }
        stopTimer();

      } else {
        getCout() << "\n  DOT : Unknown OMP Target tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  DOT : Unknown OMP Target variant id = " << vid << std::endl;
    }

  }

}

void DOT::setOpenMPTargetTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_OpenMPTarget) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
