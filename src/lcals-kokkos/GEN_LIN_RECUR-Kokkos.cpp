//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void GEN_LIN_RECUR::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize();

  GEN_LIN_RECUR_DATA_SETUP;

// wrap pointers in Kokkos Views

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

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //   NOTA BENE:
        //   Index_type kb5i = m_kb5i;
        //   Index_type N = m_N;

        Kokkos::parallel_for("GEN_LIN_RECUR_Kokkos Kokkos Lambda -- BODY1",
                             // Here, RAJAPerf Suite (RPS) indices are (0, N) for BODY1
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                             KOKKOS_LAMBDA(Index_type k) {
                             /*
                              * #define GEN_LIN_RECUR_BODY1
                              * b5[k+kb5i] = sa[k] + stb5[k]*sb[k];
                              * stb5[k] = b5[k+kb5i] - stb5[k];
                              * */
                             b5_view[k+kb5i] = sa_view[k] + stb5_view[k]*sb_view[k];
                             stb5_view[k] = b5_view[k+kb5i] - stb5_view[k];
                             });
       


        Kokkos::parallel_for("GEN_LIN_RECUR_Kokkos Kokkos Lambda -- BODY2",
                             // ATTN:  you must adjust indices to align with
                             // RPS design intent here;
                             // RPS indices are (1, N+1) for BODY2
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(1, N+1),
                             KOKKOS_LAMBDA(Index_type i) {
                             /*
                             #define GEN_LIN_RECUR_BODY2  \
                             Index_type k = N - i ; \
                             b5[k+kb5i] = sa[k] + stb5[k]*sb[k]; \
                             stb5[k] = b5[k+kb5i] - stb5[k];
                             */
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

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(b5, b5_view, iend);
  moveDataToHostFromKokkosView(sa, sa_view, iend);
  moveDataToHostFromKokkosView(sb, sb_view, iend);
  moveDataToHostFromKokkosView(stb5, stb5_view, iend);

}

} // end namespace lcals
} // end namespace rajaperf
