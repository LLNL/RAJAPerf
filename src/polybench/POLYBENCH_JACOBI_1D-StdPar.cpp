//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_JACOBI_1D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  counting_iterator<Index_type> begin(1);
  counting_iterator<Index_type> end(N-1);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

          std::for_each( std::execution::par_unseq,
                         begin, end,
                         [=](Index_type i) {
            POLYBENCH_JACOBI_1D_BODY1;
          });
          std::for_each( std::execution::par_unseq,
                         begin, end,
                         [=](Index_type i) {
            POLYBENCH_JACOBI_1D_BODY2;
          });

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }

    case Lambda_StdPar : {

      auto poly_jacobi1d_lam1 = [=] (Index_type i) {
                                  POLYBENCH_JACOBI_1D_BODY1;
                                };
      auto poly_jacobi1d_lam2 = [=] (Index_type i) {
                                  POLYBENCH_JACOBI_1D_BODY2;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          std::for_each( std::execution::par_unseq,
                         begin, end,
                          [=](Index_type i) {
            poly_jacobi1d_lam1(i);
          });
          std::for_each( std::execution::par_unseq,
                         begin, end,
                          [=](Index_type i) {
            poly_jacobi1d_lam2(i);
          });

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::forall<RAJA::loop_exec> ( RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam1
          );

          RAJA::forall<RAJA::loop_exec> ( RAJA::RangeSegment{1, N-1}, 
            poly_jacobi1d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  POLYBENCH_JACOBI_1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
