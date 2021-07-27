//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <cmath>

namespace rajaperf 
{
namespace lcals
{


void PLANCKIAN::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PLANCKIAN_DATA_SETUP;

  /*
#define PLANCKIAN_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr u = m_u; \
  Real_ptr v = m_v; \
  Real_ptr w = m_w;
*/

  auto x_view = getViewFromPointer(x, iend);
  auto y_view = getViewFromPointer(y, iend);
  auto u_view = getViewFromPointer(u, iend);
  auto v_view = getViewFromPointer(v, iend);
  auto w_view = getViewFromPointer(w, iend);


  auto planckian_lam = [=](Index_type i) {
                         PLANCKIAN_BODY;
                       };

# if defined (RUN_KOKKOS)
  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          PLANCKIAN_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          planckian_lam(i);
        }

      }
      stopTimer();

      break;
    }
/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), planckian_lam);

      }
      stopTimer();

      break;
    }
*/


    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("PLANCKIAN_Kokkos Kokkos_Lambda",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                             KOKKOS_LAMBDA(Index_type i){
                             /* #define PLANCKIAN_BODY  \
                              * y[i] = u[i] / v[i]; \
                              * w[i] = x[i] / ( exp( y[i] ) - 1.0 );
                              */
                              y_view[i] = u_view[i] / v_view[i];
                              w_view[i] = x_view[i] / ( exp( y_view[i] ) - 1.0 );
                             });
      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  PLANCKIAN : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(x, x_view, iend);
  moveDataToHostFromKokkosView(y, y_view, iend);
  moveDataToHostFromKokkosView(u, u_view, iend);
  moveDataToHostFromKokkosView(v, v_view, iend);
  moveDataToHostFromKokkosView(w, w_view, iend);



}

} // end namespace lcals
} // end namespace rajaperf
