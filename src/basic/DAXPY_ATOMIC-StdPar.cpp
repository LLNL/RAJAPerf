//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>
#include <atomic>

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

  DAXPY_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
#if defined(NVCXX_GPU_ENABLED)
          //atomicAdd(&y[i],a * x[i]);
          atomicaddd(&y[i],a * x[i]);
#elif defined(_OPENMP)
          #pragma omp atomic
          y[i] += a * x[i];
#elif defined(_OPENACC)
          #pragma acc atomic
          y[i] += a * x[i];
#elif __cpp_lib_atomic_ref
          auto px = std::atomic_ref<Real_type>(x[i]);
          auto py = std::atomic_ref<Real_type>(y[i]);
          py += a * px;
#else
#warning No atomic
          y[i] += a * x[i];
#endif
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto daxpy_atomic_lam = [=](Index_type i) {
#if defined(NVCXX_GPU_ENABLED)
          //atomicAdd(&y[i],a * x[i]);
          atomicaddd(&y[i],a * x[i]);
#elif defined(_OPENMP)
          #pragma omp atomic
          y[i] += a * x[i];
#elif defined(_OPENACC)
          #pragma acc atomic
          y[i] += a * x[i];
#elif __cpp_lib_atomic_ref
          auto px = std::atomic_ref<Real_type>(x[i]);
          auto py = std::atomic_ref<Real_type>(y[i]);
          py += a * px;
#else
#warning No atomic
          y[i] += a * x[i];
#endif
      };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
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

#endif  // RAJA_ENABLE_STDPAR

