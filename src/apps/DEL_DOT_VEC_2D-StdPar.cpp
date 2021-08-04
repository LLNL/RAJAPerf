//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include "AppsData.hpp"

#include "camp/resource.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void DEL_DOT_VEC_2D::runStdParVariant(VariantID vid)
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

      auto range = std::views::iota(ibegin,iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
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

      auto range = std::views::iota(ibegin,iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
                        [=](Index_type ii) {
          deldotvec2d_base_lam(ii);
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      camp::resources::Resource working_res{camp::resources::Host()};
      RAJA::TypedListSegment<Index_type> zones(m_domain->real_zones, 
                                               m_domain->n_real_zones, 
                                               working_res);

      auto deldotvec2d_lam = [=](Index_type i) {
                               DEL_DOT_VEC_2D_BODY;
                             };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(zones, deldotvec2d_lam);

      }
      stopTimer(); 

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  DEL_DOT_VEC_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf
