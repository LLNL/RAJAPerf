//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void INT_PREDICT::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INT_PREDICT_DATA_SETUP;

  // Wrap pointer in Kokkos View, and adjust indices
  auto px_view = getViewFromPointer(px, iend * 13);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INT_PREDICT_1");
      Kokkos::parallel_for(
          "INT_PREDICT_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            px_view[i] =
                dm28 * px_view[i + offset * 12] +
                dm27 * px_view[i + offset * 11] +
                dm26 * px_view[i + offset * 10] +
                dm25 * px_view[i + offset * 9] +
                dm24 * px_view[i + offset * 8] +
                dm23 * px_view[i + offset * 7] +
                dm22 * px_view[i + offset * 6] +
                c0 * (px_view[i + offset * 4] + px_view[i + offset * 5]) +
                px_view[i + offset * 2];
          });
      RP_CALI_SUBKERNEL_END("INT_PREDICT_1");
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  INT_PREDICT : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(px, px_view, iend * 13);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INT_PREDICT, Kokkos, Kokkos_Lambda)

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
