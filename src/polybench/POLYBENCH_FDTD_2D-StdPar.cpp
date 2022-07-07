//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_FDTD_2D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  counting_iterator<Index_type> beginX(0);
  counting_iterator<Index_type> endX(nx);
  counting_iterator<Index_type> beginY(0);
  counting_iterator<Index_type> endY(ny);
  counting_iterator<Index_type> begin1X(1);
  counting_iterator<Index_type> end1X(nx);
  counting_iterator<Index_type> beginXm1(0);
  counting_iterator<Index_type> endXm1(nx-1);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) {

          std::for_each( std::execution::par_unseq,
                         beginY, endY,
                         [=](Index_type j) {
            POLYBENCH_FDTD_2D_BODY1;
          });
          std::for_each( std::execution::par_unseq,
                         begin1X, end1X,
                         [=](Index_type i) {
            for (Index_type j = 0; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY2;
            }
          });
          std::for_each( std::execution::par_unseq,
                         beginX, endX,
                         [=](Index_type i) {
            for (Index_type j = 1; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY3;
            }
          });
          std::for_each( std::execution::par_unseq,
                         beginXm1, endXm1,
                         [=](Index_type i) {
            for (Index_type j = 0; j < ny - 1; j++) {
              POLYBENCH_FDTD_2D_BODY4;
            }
          });

        }  // tstep loop

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      //
      // Note: first lambda must use capture by reference so that the
      //       scalar variable 't' used in it is updated for each
      //       t-loop iteration.
      //
      auto poly_fdtd2d_base_lam1 = [&](Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY1;
                                   };
      auto poly_fdtd2d_base_lam2 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY2;
                                   };
      auto poly_fdtd2d_base_lam3 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY3;
                                   };
      auto poly_fdtd2d_base_lam4 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY4;
                                   };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) {

          std::for_each( //std::execution::par_unseq,
                         beginY, endY,
                         [=](Index_type j) {
            poly_fdtd2d_base_lam1(j);
          });
          std::for_each( //std::execution::par_unseq,
                         begin1X, end1X,
                         [=](Index_type i) {
            for (Index_type j = 0; j < ny; j++) {
              poly_fdtd2d_base_lam2(i, j);
            }
          });
          std::for_each( //std::execution::par_unseq,
                         beginX, endX,
                         [=](Index_type i) {
            for (Index_type j = 1; j < ny; j++) {
              poly_fdtd2d_base_lam3(i, j);
            }
          });
          std::for_each( //std::execution::par_unseq,
                         beginXm1, endXm1,
                         [=](Index_type i) {
            for (Index_type j = 0; j < ny - 1; j++) {
              poly_fdtd2d_base_lam4(i, j);
            }
          });

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      POLYBENCH_FDTD_2D_VIEWS_RAJA;

      //
      // Note: first lambda must use capture by reference so that the
      //       scalar variable 't' used in it is updated for each
      //       t-loop iteration.
      //
      auto poly_fdtd2d_lam1 = [&](Index_type j) {
                                POLYBENCH_FDTD_2D_BODY1_RAJA;
                              };
      auto poly_fdtd2d_lam2 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY2_RAJA;
                              };
      auto poly_fdtd2d_lam3 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY3_RAJA;
                              };
      auto poly_fdtd2d_lam4 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY4_RAJA;
                              };

      using EXEC_POL1 = RAJA::loop_exec;

      using EXEC_POL234 =  
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) { 

          RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny), 
            poly_fdtd2d_lam1
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                             RAJA::RangeSegment{0, ny}),
            poly_fdtd2d_lam2
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                             RAJA::RangeSegment{1, ny}),
            poly_fdtd2d_lam3
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                             RAJA::RangeSegment{0, ny-1}),
            poly_fdtd2d_lam4
          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }

#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\nPOLYBENCH_FDTD_2D  Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
