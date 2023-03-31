//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#ifndef _OPENMP
#error Currently, OpenMP atomics are required here.
#endif

#if defined(__NVCOMPILER_CUDA__) || defined(_NVHPC_STDPAR_CUDA)
#include <cuda/atomic>
typedef cuda::std::atomic<double> myAtomic;
#else
// .fetch_add() for double is not available yet...
#include <atomic>
typedef std::atomic<double> myAtomic;
#endif

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void PI_ATOMIC::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //myAtomic a_pi{m_pi_init};
        *pi = m_pi_init;
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          double x = (double(i) + 0.5) * dx;
          _Pragma("omp atomic")
          *pi += dx / (1.0 + x * x);
          //a_pi.fetch_add(dx / (1.0 + x * x));
        });
        //*pi = a_pi * 4.0;
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto piatomic_base_lam = [=](Index_type i) {
                                 double x = (double(i) + 0.5) * dx;
                                 _Pragma("omp atomic")
                                 *pi += dx / (1.0 + x * x);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                       [=](Index_type i) {
          piatomic_base_lam(i);
        });
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  PI_ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

