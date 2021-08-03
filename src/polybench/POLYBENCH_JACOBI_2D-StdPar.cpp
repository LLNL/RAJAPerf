//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace polybench
{


void POLYBENCH_JACOBI_2D::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota((Index_type)1,N-1);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

          std::for_each( std::execution::par_unseq,
                          std::begin(range), std::end(range),
                          [=](Index_type i) {
            std::for_each( std::execution::unseq,
                            std::begin(range), std::end(range),
                            [=](Index_type j) {
              POLYBENCH_JACOBI_2D_BODY1;
            });
          });
          std::for_each( std::execution::par_unseq,
                          std::begin(range), std::end(range),
                          [=](Index_type i) {
            std::for_each( std::execution::unseq,
                            std::begin(range), std::end(range),
                            [=](Index_type j) {
              POLYBENCH_JACOBI_2D_BODY2;
            });
          });

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }


    case Lambda_StdPar : {

      auto poly_jacobi2d_base_lam1 = [=](Index_type i, Index_type j) {
                                       POLYBENCH_JACOBI_2D_BODY1;
                                     };
      auto poly_jacobi2d_base_lam2 = [=](Index_type i, Index_type j) {
                                       POLYBENCH_JACOBI_2D_BODY2;
                                     };

      auto range = std::views::iota((Index_type)1,N-1);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          std::for_each( std::execution::par_unseq,
                          std::begin(range), std::end(range),
                          [=](Index_type i) {
            std::for_each( std::execution::unseq,
                            std::begin(range), std::end(range),
                            [=](Index_type j) {
              poly_jacobi2d_base_lam1(i, j);
            });
          });

          std::for_each( std::execution::par_unseq,
                          std::begin(range), std::end(range),
                          [=](Index_type i) {
            std::for_each( std::execution::unseq,
                            std::begin(range), std::end(range),
                            [=](Index_type j) {
              poly_jacobi2d_base_lam2(i, j);
            });
          });

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      POLYBENCH_JACOBI_2D_VIEWS_RAJA;

      auto poly_jacobi2d_lam1 = [=](Index_type i, Index_type j) {
                                  POLYBENCH_JACOBI_2D_BODY1_RAJA;
                                };
      auto poly_jacobi2d_lam2 = [=](Index_type i, Index_type j) {
                                  POLYBENCH_JACOBI_2D_BODY2_RAJA;
                                };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >,
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

            poly_jacobi2d_lam1,
            poly_jacobi2d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  POLYBENCH_JACOBI_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
