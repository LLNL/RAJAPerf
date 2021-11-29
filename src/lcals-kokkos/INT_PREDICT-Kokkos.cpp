//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void INT_PREDICT::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INT_PREDICT_DATA_SETUP;

/*
   *#define INT_PREDICT_DATA_SETUP \
  Real_ptr px = m_px; \
  Real_type dm22 = m_dm22; \
  Real_type dm23 = m_dm23; \
  Real_type dm24 = m_dm24; \
  Real_type dm25 = m_dm25; \
  Real_type dm26 = m_dm26; \
  Real_type dm27 = m_dm27; \
  Real_type dm28 = m_dm28; \
  Real_type c0 = m_c0; \

*/

  // Wrap pointer in Kokkos View, and adjust indices
  auto px_view = getViewFromPointer(px, iend*13);


  auto intpredict_lam = [=](Index_type i) {
                          INT_PREDICT_BODY;
                        };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        // Declare variables in INT_PREDICT.hpp
        Real_type dm22 = m_dm22;
        Real_type dm23 = m_dm23;
        Real_type dm24 = m_dm24;
        Real_type dm25 = m_dm25;
        Real_type dm26 = m_dm26;
        Real_type dm27 = m_dm27;
        Real_type dm28 = m_dm28;

        /*
        #define INT_PREDICT_BODY  \
        px[i] = dm28*px[i + offset * 12] + dm27*px[i + offset * 11] + \
        dm26*px[i + offset * 10] + dm25*px[i + offset *  9] + \
        dm24*px[i + offset *  8] + dm23*px[i + offset *  7] + \
        dm22*px[i + offset *  6] + \
        c0*( px[i + offset *  4] + px[i + offset *  5] ) + \
        px[i + offset *  2];
        */
        Kokkos::parallel_for("INT_PREDICT_Kokkos Kokkos_Lambda",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                             KOKKOS_LAMBDA(Index_type i){
                             // #define INT_PREDICT_BODY
                             px_view[i] = dm28*px_view[i + offset * 12] + dm27*px_view[i + offset * 11] + \
                             dm26*px_view[i + offset * 10] + dm25*px_view[i + offset *  9] + \
                             dm24*px_view[i + offset *  8] + dm23*px_view[i + offset *  7] + \
                             dm22*px_view[i + offset *  6] + \
                             c0*( px_view[i + offset *  4] + px_view[i + offset *  5] ) + \
                             px_view[i + offset *  2];
                             });

      }
      Kokkos::fence();
      stopTimer();

      break;
    }


    default : {
      std::cout << "\n  INT_PREDICT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

    moveDataToHostFromKokkosView(px, px_view, iend*13);
}

} // end namespace lcals
} // end namespace rajaperf
