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

#ifdef USE_STDPAR_COLLAPSE
  const auto nn = N-2;
  counting_iterator<Index_type> begin(0);
  counting_iterator<Index_type> end(nn*nn);
#else
  counting_iterator<Index_type> begin(1);
  counting_iterator<Index_type> end(N-1);
#endif

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type ij) {
              const auto i  = 1 + ij / nn;
              const auto j  = 1 + ij % nn;
#else
          std::for_each( std::execution::par_unseq,
                         begin, end,
                         [=](Index_type i) {
            std::for_each( std::execution::unseq,
                           begin, end,
                           [=](Index_type j) {
#endif
              POLYBENCH_JACOBI_2D_BODY1;
#ifndef USE_STDPAR_COLLAPSE
            });
#endif
          });
#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type ij) {
              const auto i  = 1 + ij / nn;
              const auto j  = 1 + ij % nn;
#else
          std::for_each( std::execution::par_unseq,
                         begin, end,
                         [=](Index_type i) {
            std::for_each( std::execution::unseq,
                           begin, end,
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
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type ij) {
              const auto i  = 1 + ij / nn;
              const auto j  = 1 + ij % nn;
#else
          std::for_each( std::execution::par_unseq,
                         begin, end,
                          [=](Index_type i) {
            std::for_each( std::execution::unseq,
                           begin, end,
                            [=](Index_type j) {
#endif
              poly_jacobi2d_base_lam1(i, j);
#ifndef USE_STDPAR_COLLAPSE
            });
#endif
          });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type ij) {
              const auto i  = 1 + ij / nn;
              const auto j  = 1 + ij % nn;
#else
          std::for_each( std::execution::par_unseq,
                         begin, end,
                          [=](Index_type i) {
            std::for_each( std::execution::unseq,
                           begin, end,
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
