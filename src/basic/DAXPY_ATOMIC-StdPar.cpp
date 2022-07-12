//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>
#include <atomic>

#if defined(NVCXX_GPU_ENABLED)
// this is required to get NVC++ to compile CUDA atomics in StdPar
#include <openacc.h>
#endif

namespace rajaperf
{
namespace basic
{


void DAXPY_ATOMIC::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  DAXPY_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin, end,
                       [=](Index_type i) {
#if __cpp_lib_atomic_ref
          auto px = std::atomic_ref<Real_type>(&x[i]);
          auto py = std::atomic_ref<Real_type>(&y[i]);
          py += a * px;
#elif defined(_OPENMP)
          #pragma omp atomic
          y[i] += a * x[i];
#elif defined(_OPENACC)
          #pragma acc atomic
          y[i] += a * x[i];
#elif defined(NVCXX_GPU_ENABLED)
          atomicaddd(&y[i],a * x[i]);
#else
#error No atomic
#endif
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto daxpy_atomic_lam = [=](Index_type i) {
                         #pragma omp atomic
                         y[i] += a * x[i] ;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin, end,
                       [=](Index_type i) {
          daxpy_atomic_lam(i);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  DAXPY_ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
