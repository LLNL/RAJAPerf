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
            std::for_each( std::execution::unseq,
                           begin, end,
                           [=](Index_type j) {
              std::for_each( std::execution::unseq,
                           begin, end,
                           [=](Index_type k) {
                POLYBENCH_HEAT_3D_BODY1;
              });
            });
          });

          std::for_each( std::execution::par_unseq,
                         begin, end,
                         [=](Index_type i) {
            std::for_each( std::execution::unseq,
                           begin, end,
                           [=](Index_type j) {
              std::for_each( std::execution::unseq,
                           begin, end,
                           [=](Index_type k) {
                POLYBENCH_HEAT_3D_BODY2;
              });
            });
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

          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              for (Index_type k = 1; k < N-1; ++k ) {
                poly_heat3d_base_lam1(i, j, k);
              }
            }
          }

          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              for (Index_type k = 1; k < N-1; ++k ) {
                poly_heat3d_base_lam2(i, j, k);
              }
            }
          }

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
