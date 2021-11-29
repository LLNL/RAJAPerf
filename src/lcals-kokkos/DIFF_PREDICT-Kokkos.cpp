//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

template<class px_type, class cx_type>
void diff_predict_helper(Index_type run_reps,
                         Index_type ibegin,
                         Index_type iend,
                         Index_type offset,
                         // a Kokkos View
                         px_type& px, 
                         // a Kokkos View
                         cx_type& cx){

      
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
             
              Kokkos::parallel_for("DIFF_PREDICT_Kokkos Kokkos_Lambda",
                                   Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                   KOKKOS_LAMBDA(Index_type i) {
                                   DIFF_PREDICT_BODY
                                   });

      }
}



void DIFF_PREDICT::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();


  DIFF_PREDICT_DATA_SETUP;


  // Instiating KokkosViews using getViewFromPointer;
  // Wrapping pointers in KokkosViews

  // You need to know the actual array size here to catch errors;
  
  auto px_view = getViewFromPointer(px, iend*14);
  auto cx_view = getViewFromPointer(cx, iend*14);
  

  auto diffpredict_lam = [=](Index_type i) {
                           DIFF_PREDICT_BODY;
                         };

  #if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {
      Kokkos::fence();
      startTimer();

	  diff_predict_helper(run_reps,
                          ibegin,
                          iend,
                          offset,
                          px_view,
                          cx_view);


      Kokkos::fence();
      stopTimer();
      break;

    }


    default : {
      std::cout << "\n  DIFF_PREDICT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(px, px_view, iend*14);
  moveDataToHostFromKokkosView(cx, cx_view, iend*14);

}

} // end namespace lcals
} // end namespace rajaperf
