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

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) {

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), ny,
                           [=](Index_type j) {
            POLYBENCH_FDTD_2D_BODY1;
          });

          // Note to future developers:
          //   Do not try to be smart and use more C++ than necessary.
          //   auto [i,j] = std::div(ij,ny); i++;
          //   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ This is noticeably slower than below.

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), (nx-1)*ny,
                           [=](Index_type ij) {
              const auto i  = 1 + ij / ny;
              const auto j  =     ij % ny;
              POLYBENCH_FDTD_2D_BODY2;
          });

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), nx*(ny-1),
                           [=](Index_type ij) {
              const auto i  =     ij / (ny-1);
              const auto j  = 1 + ij % (ny-1);
              POLYBENCH_FDTD_2D_BODY3;
          });

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), (nx-1)*(ny-1),
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
      // capturing t by reference is required for GCC 11 to generate correct results
      // but that breaks NVHPC GPU, so we instead make it an explicit parameter
      auto poly_fdtd2d_base_lam1 = [=](Index_type j, Index_type t) {
                                     //ey[j + 0*ny] = fict[t];
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

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), ny,
                           [=](Index_type j) {
            poly_fdtd2d_base_lam1(j,t);
          });

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), (nx-1)*ny,
                           [=](Index_type ij) {
              const auto i  = 1 + ij / ny;
              const auto j  =     ij % ny;
              poly_fdtd2d_base_lam2(i, j);
          });

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), nx*(ny-1),
                           [=](Index_type ij) {
              const auto i  =     ij / (ny-1);
              const auto j  = 1 + ij % (ny-1);
              poly_fdtd2d_base_lam3(i, j);
          });

          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), (nx-1)*(ny-1),
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
