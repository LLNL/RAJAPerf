//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void HYDRO_2D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  auto beginK = counting_iterator<Index_type>(kbeg);
  auto endK   = counting_iterator<Index_type>(kend);
  auto beginJ = counting_iterator<Index_type>(jbeg);
  auto endJ   = counting_iterator<Index_type>(jend);

  HYDRO_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par,
                        beginK, endK,
                        [=](Index_type k) {
          std::for_each( std::execution::unseq,
                        beginJ, endJ,
                        [=](Index_type j) {
            HYDRO_2D_BODY1;
          });
        });

        std::for_each( std::execution::par,
                        beginK, endK,
                        [=](Index_type k) {
          std::for_each( std::execution::unseq,
                        beginJ, endJ,
                        [=](Index_type j) {
            HYDRO_2D_BODY2;
          });
        });

        std::for_each( std::execution::par,
                        beginK, endK,
                        [=](Index_type k) {
          std::for_each( std::execution::unseq,
                        beginJ, endJ,
                        [=](Index_type j) {
            HYDRO_2D_BODY3;
          });
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto hydro2d_base_lam1 = [=] (Index_type k, Index_type j) {
                                 HYDRO_2D_BODY1;
                               };
      auto hydro2d_base_lam2 = [=] (Index_type k, Index_type j) {
                                 HYDRO_2D_BODY2;
                               };
      auto hydro2d_base_lam3 = [=] (Index_type k, Index_type j) {
                                 HYDRO_2D_BODY3;
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par,
                        beginK, endK,
                        [=](Index_type k) {
          std::for_each( std::execution::unseq,
                        beginJ, endJ,
                        [=](Index_type j) {
            hydro2d_base_lam1(k, j);
          });
        });

        std::for_each( std::execution::par,
                        beginK, endK,
                        [=](Index_type k) {
          std::for_each( std::execution::unseq,
                        beginJ, endJ,
                        [=](Index_type j) {
            hydro2d_base_lam2(k, j);
          });
        });

        std::for_each( std::execution::par,
                        beginK, endK,
                        [=](Index_type k) {
          std::for_each( std::execution::unseq,
                        beginJ, endJ,
                        [=](Index_type j) {
            hydro2d_base_lam3(k, j);
          });
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
