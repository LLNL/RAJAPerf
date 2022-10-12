//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_JACOBI_2D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  const auto n2 = (N-2);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n2*n2,
                         [=](Index_type ij) {
              const auto i  = 1 + ij / n2;
              const auto j  = 1 + ij % n2;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), n2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), n2,
                             [=](Index_type j) {
#endif
              POLYBENCH_JACOBI_2D_BODY1;
#ifndef USE_STDPAR_COLLAPSE
            });
#endif
          });
#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n2*n2,
                         [=](Index_type ij) {
              const auto i  = 1 + ij / n2;
              const auto j  = 1 + ij % n2;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), n2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), n2,
                             [=](Index_type j) {
#endif
              POLYBENCH_JACOBI_2D_BODY2;
#ifndef USE_STDPAR_COLLAPSE
            });
#endif
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

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n2*n2,
                         [=](Index_type ij) {
              const auto i  = 1 + ij / n2;
              const auto j  = 1 + ij % n2;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), n2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), n2,
                             [=](Index_type j) {
#endif
              poly_jacobi2d_base_lam1(i, j);
#ifndef USE_STDPAR_COLLAPSE
            });
#endif
          });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n2*n2,
                         [=](Index_type ij) {
              const auto i  = 1 + ij / n2;
              const auto j  = 1 + ij % n2;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), n2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), n2,
                             [=](Index_type j) {
#endif
              poly_jacobi2d_base_lam2(i, j);
#ifndef USE_STDPAR_COLLAPSE
            });
#endif
          });

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
