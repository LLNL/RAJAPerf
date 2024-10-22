//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void PI_ATOMIC::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          double x = (double(i) + 0.5) * dx;
          #pragma omp atomic
          *pi += dx / (1.0 + x * x);
        }
        m_pi_final = *pi * 4.0;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto piatomic_base_lam = [=](Index_type i) {
                                 double x = (double(i) + 0.5) * dx;
                                 #pragma omp atomic
                                 *pi += dx / (1.0 + x * x);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          piatomic_base_lam(i);
        }
        m_pi_final = *pi * 4.0;

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            double x = (double(i) + 0.5) * dx;
            RAJA::atomicAdd<RAJA::omp_atomic>(pi, dx / (1.0 + x * x));
        });
        m_pi_final = *pi * 4.0;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  PI_ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

  PI_ATOMIC_DATA_TEARDOWN;

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
