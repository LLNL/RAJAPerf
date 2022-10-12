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

  counting_iterator<Index_type> beginY(0);
  counting_iterator<Index_type> endY(ny);

  counting_iterator<Index_type> begin2(0);
  counting_iterator<Index_type> end2((nx-1)*ny);

  counting_iterator<Index_type> begin3(0);
  counting_iterator<Index_type> end3(nx*(ny-1));

  counting_iterator<Index_type> begin4(0);
  counting_iterator<Index_type> end4((nx-1)*(ny-1));

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
                         begin2, end2,
                         [=](Index_type ij) {
              const auto i  = 1 + ij / ny;
              const auto j  =     ij % ny;
              POLYBENCH_FDTD_2D_BODY2;
          });

          std::for_each( std::execution::par_unseq,
                         begin3, end3,
                         [=](Index_type ij) {
              const auto i  =     ij / (ny-1);
              const auto j  = 1 + ij % (ny-1);
              POLYBENCH_FDTD_2D_BODY3;
          });

          std::for_each( std::execution::par_unseq,
                         begin4, end4,
                         [=](Index_type ij) {
              const auto i  = ij / (ny-1);
              const auto j  = ij % (ny-1);
              POLYBENCH_FDTD_2D_BODY4;
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
      // THIS ONE, AND ONLY THIS ONE, NEEDS TO BE [&] TO BE CORRECT, AT LEAST WITH GCC
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

          std::for_each( std::execution::par_unseq,
                         beginY, endY,
                         [=](Index_type j) {
            poly_fdtd2d_base_lam1(j);
          });

          std::for_each( std::execution::par_unseq,
                         begin2, end2,
                         [=](Index_type ij) {
              const auto i  = 1 + ij / ny;
              const auto j  =     ij % ny;
              poly_fdtd2d_base_lam2(i, j);
          });

          std::for_each( std::execution::par_unseq,
                         begin3, end3,
                         [=](Index_type ij) {
              const auto i  =     ij / (ny-1);
              const auto j  = 1 + ij % (ny-1);
              poly_fdtd2d_base_lam3(i, j);
          });

          counting_iterator<Index_type> begin4(0);
          counting_iterator<Index_type> end4((nx-1)*(ny-1));

          std::for_each( std::execution::par_unseq,
                         begin4, end4,
                         [=](Index_type ij) {
              const auto i  = ij / (ny-1);
              const auto j  = ij % (ny-1);
              poly_fdtd2d_base_lam4(i, j);
          });

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    default : {
      getCout() << "\nPOLYBENCH_FDTD_2D  Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
