//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void GEN_LIN_RECUR::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  GEN_LIN_RECUR_DATA_SETUP;

  // Wrap pointers in Kokkos Views

  auto b5_view = getViewFromPointer(b5, iend);
  auto sa_view = getViewFromPointer(sa, iend);
  auto sb_view = getViewFromPointer(sb, iend);
  auto stb5_view = getViewFromPointer(stb5, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
      Kokkos::parallel_for(
          "GEN_LIN_RECUR_Kokkos Kokkos Lambda -- BODY1",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
          KOKKOS_LAMBDA(Index_type k) {
            b5_view[k + kb5i] = sa_view[k] + stb5_view[k] * sb_view[k];
            stb5_view[k] = b5_view[k + kb5i] - stb5_view[k];
          });
      RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

      RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
      Kokkos::parallel_for(
          "GEN_LIN_RECUR_Kokkos Kokkos Lambda -- BODY2",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(1, N + 1),
          KOKKOS_LAMBDA(Index_type i) {
            Index_type k = N - i;

            b5_view[k + kb5i] = sa_view[k] + stb5_view[k] * sb_view[k];
            stb5_view[k] = b5_view[k + kb5i] - stb5_view[k];
          });
      RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid
              << std::endl;
  }
  }

  moveDataToHostFromKokkosView(b5, b5_view, iend);
  moveDataToHostFromKokkosView(sa, sa_view, iend);
  moveDataToHostFromKokkosView(sb, sb_view, iend);
  moveDataToHostFromKokkosView(stb5, stb5_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(GEN_LIN_RECUR, Kokkos, Kokkos_Lambda)

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
