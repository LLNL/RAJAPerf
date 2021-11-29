//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void TRIDIAG_ELIM::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

/*
#define TRIDIAG_ELIM_DATA_SETUP \
  Real_ptr xout = m_xout; \
  Real_ptr xin = m_xin; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z;
*/


  auto xout_view = getViewFromPointer(xout, iend);
  auto xin_view  = getViewFromPointer(xin, iend);
  auto y_view    = getViewFromPointer(y, iend);
  auto z_view    = getViewFromPointer(z, iend);


  auto tridiag_elim_lam = [=](Index_type i) {
                            TRIDIAG_ELIM_BODY;
                          };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

      Kokkos::fence();   
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        
        Kokkos::parallel_for("TRIDIAG_ELIM_Kokkos Kokkos_Lambda",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                             KOKKOS_LAMBDA(Index_type i){
                             // #define TRIDIAG_ELIM_BODY
                             // xout[i] = z[i] * ( y[i] - xin[i-1] );
                             xout_view[i] = z_view[i] * ( y_view[i] - xin_view[i-1] );
                             });
      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  TRIDIAG_ELIM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(xout, xout_view, iend);
  moveDataToHostFromKokkosView(xin, xin_view, iend);
  moveDataToHostFromKokkosView(y, y_view, iend);
  moveDataToHostFromKokkosView(z, z_view, iend);

}

} // end namespace lcals
} // end namespace rajaperf
