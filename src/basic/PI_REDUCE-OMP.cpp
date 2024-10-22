//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void PI_REDUCE::runOpenMPVariant(VariantID vid, size_t tune_idx)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type pi = m_pi_init;

        #pragma omp parallel for reduction(+:pi)
        for (Index_type i = ibegin; i < iend; ++i ) {
          PI_REDUCE_BODY;
        }

        m_pi = 4.0 * pi;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto pireduce_base_lam = [=](Index_type i) -> Real_type {
                                 double x = (double(i) + 0.5) * dx;
                                 return dx / (1.0 + x * x);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type pi = m_pi_init;

        #pragma omp parallel for reduction(+:pi)
        for (Index_type i = ibegin; i < iend; ++i ) {
          pi += pireduce_base_lam(i);
        }

        m_pi = 4.0 * pi;

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      if (tune_idx == 0) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          RAJA::ReduceSum<RAJA::omp_reduce, Real_type> pi(m_pi_init);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            [=](Index_type i) {
              PI_REDUCE_BODY;
          });

          m_pi = 4.0 * pi.get();

        }
        stopTimer();

      } else if (tune_idx == 1) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          Real_type tpi = m_pi_init;

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tpi),
            [=] (Index_type i, Real_type& pi) {
              PI_REDUCE_BODY;
            }
          );

          m_pi = static_cast<Real_type>(tpi) * 4.0;

        }
        stopTimer();

      } else {
        getCout() << "\n  PI_REDUCE : Unknown OpenMP tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  PI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
  RAJA_UNUSED_VAR(tune_idx);
#endif
}

void PI_REDUCE::setOpenMPTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_OpenMP) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace basic
} // end namespace rajaperf
