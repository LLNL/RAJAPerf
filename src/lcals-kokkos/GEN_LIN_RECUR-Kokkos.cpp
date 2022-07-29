//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void GEN_LIN_RECUR::runKokkosVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize();

  GEN_LIN_RECUR_DATA_SETUP;

// Wrap pointers in Kokkos Views

  auto b5_view = getViewFromPointer(b5, iend);
  auto sa_view = getViewFromPointer(sa, iend);
  auto sb_view = getViewFromPointer(sb, iend);
  auto stb5_view = getViewFromPointer(stb5, iend);

// RAJAPerf Suite Lambdas

  auto genlinrecur_lam1 = [=](Index_type k) {
                            GEN_LIN_RECUR_BODY1;
                          };
  auto genlinrecur_lam2 = [=](Index_type i) {
                            GEN_LIN_RECUR_BODY2;
                          };

  switch ( vid ) {

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("GEN_LIN_RECUR_Kokkos Kokkos Lambda -- BODY1",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                             KOKKOS_LAMBDA(Index_type k) {
                             b5_view[k+kb5i] = sa_view[k] + stb5_view[k]*sb_view[k];
                             stb5_view[k] = b5_view[k+kb5i] - stb5_view[k];
                             });
       


        Kokkos::parallel_for("GEN_LIN_RECUR_Kokkos Kokkos Lambda -- BODY2",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(1, N+1),
                             KOKKOS_LAMBDA(Index_type i) {
                             Index_type k = N - i ;

                             b5_view[k+kb5i] = sa_view[k] + stb5_view[k]*sb_view[k];
                             stb5_view[k] = b5_view[k+kb5i] - stb5_view[k];

                             });
                             
      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }


  moveDataToHostFromKokkosView(b5, b5_view, iend);
  moveDataToHostFromKokkosView(sa, sa_view, iend);
  moveDataToHostFromKokkosView(sb, sb_view, iend);
  moveDataToHostFromKokkosView(stb5, stb5_view, iend);

}

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
