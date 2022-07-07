//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(__NVCOMPILER_CUDA__) || defined(_NVHPC_STDPAR_CUDA)
#include <cuda/atomic>
typedef cuda::std::atomic<double> myAtomic;
#else
#include <atomic>
typedef std::atomic<double> myAtomic;
#endif

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

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

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  PI_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        myAtomic a_pi{m_pi_init};
        std::for_each( std::execution::par_unseq,
                       begin, end,
                       [=,&a_pi](Index_type i) {
          double x = (double(i) + 0.5) * dx;
          a_pi = a_pi + dx / (1.0 + x * x);
        });
        *pi = a_pi * 4.0;

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto piatomic_base_lam = [=](Index_type i, myAtomic &a_pi) {
                                 double x = (double(i) + 0.5) * dx;
                                 a_pi = a_pi + dx / (1.0 + x * x);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        myAtomic a_pi{m_pi_init};
        for (Index_type i = ibegin; i < iend; ++i ) {
          piatomic_base_lam(i,a_pi);
        }
        *pi = a_pi * 4.0;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
   
        *pi = m_pi_init;
        RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(ibegin, iend), 
          [=](Index_type i) {
            double x = (double(i) + 0.5) * dx;
            RAJA::atomicAdd<RAJA::seq_atomic>(pi, dx / (1.0 + x * x));
        });
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  PI_ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
