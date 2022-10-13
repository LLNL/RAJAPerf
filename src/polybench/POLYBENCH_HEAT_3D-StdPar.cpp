//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_HEAT_3D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

#ifdef USE_STDPAR_COLLAPSE
  const auto nn = N-2;
#endif

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nn*nn*nn,
                         [=](Index_type ijk) {
              const auto i  = 1 + ijk / (nn*nn);
              const auto jk = ijk % (nn*nn);
              const auto j  = 1 + jk / nn;
              const auto k  = 1 + jk % nn;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), N-2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), N-2,
                             [=](Index_type j) {
              std::for_each_n( std::execution::unseq,
                               counting_iterator<Index_type>(1), N-2,
                               [=](Index_type k) {
#endif
                POLYBENCH_HEAT_3D_BODY1;
#ifndef USE_STDPAR_COLLAPSE
              });
            });
#endif
          });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nn*nn*nn,
                         [=](Index_type ijk) {
              const auto i  = 1 + ijk / (nn*nn);
              const auto jk = ijk % (nn*nn);
              const auto j  = 1 + jk / nn;
              const auto k  = 1 + jk % nn;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), N-2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), N-2,
                             [=](Index_type j) {
              std::for_each_n( std::execution::unseq,
                               counting_iterator<Index_type>(1), N-2,
                               [=](Index_type k) {
#endif
                POLYBENCH_HEAT_3D_BODY2;
#ifndef USE_STDPAR_COLLAPSE
              });
            });
#endif
          });

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;

      break;
    }

    case Lambda_StdPar : {

      auto poly_heat3d_base_lam1 = [=](Index_type i, Index_type j, 
                                       Index_type k) {
                                     POLYBENCH_HEAT_3D_BODY1;
                                   };
      auto poly_heat3d_base_lam2 = [=](Index_type i, Index_type j, 
                                       Index_type k) {
                                     POLYBENCH_HEAT_3D_BODY2;
                                   };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nn*nn*nn,
                         [=](Index_type ijk) {
              const auto i  = 1 + ijk / (nn*nn);
              const auto jk = ijk % (nn*nn);
              const auto j  = 1 + jk / nn;
              const auto k  = 1 + jk % nn;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), N-2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), N-2,
                             [=](Index_type j) {
              std::for_each_n( std::execution::unseq,
                               counting_iterator<Index_type>(1), N-2,
                               [=](Index_type k) {
#endif
                poly_heat3d_base_lam1(i, j, k);
#ifndef USE_STDPAR_COLLAPSE
              });
            });
#endif
          });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nn*nn*nn,
                         [=](Index_type ijk) {
              const auto i  = 1 + ijk / (nn*nn);
              const auto jk = ijk % (nn*nn);
              const auto j  = 1 + jk / nn;
              const auto k  = 1 + jk % nn;
#else
          std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(1), N-2,
                           [=](Index_type i) {
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(1), N-2,
                             [=](Index_type j) {
              std::for_each_n( std::execution::unseq,
                               counting_iterator<Index_type>(1), N-2,
                               [=](Index_type k) {
#endif
                poly_heat3d_base_lam2(i, j, k);
#ifndef USE_STDPAR_COLLAPSE
              });
            });
#endif
          });

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_HEAT_3D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
