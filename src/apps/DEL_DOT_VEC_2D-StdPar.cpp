//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include "AppsData.hpp"

#include "camp/resource.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void DEL_DOT_VEC_2D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;


  DEL_DOT_VEC_2D_DATA_SETUP;

  NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
  NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
  NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
  NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend,
                         [=](Index_type ii) {
          DEL_DOT_VEC_2D_BODY_INDEX;
          DEL_DOT_VEC_2D_BODY;
        });

      }
      stopTimer();

      break;
    } 

    case Lambda_StdPar : {

      auto deldotvec2d_base_lam = [=](Index_type ii) {
                                    DEL_DOT_VEC_2D_BODY_INDEX;
                                    DEL_DOT_VEC_2D_BODY;
                                  };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend,
                         [=](Index_type ii) {
          deldotvec2d_base_lam(ii);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  DEL_DOT_VEC_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

