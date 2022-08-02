//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void FIRST_DIFF::runKokkosVariant(VariantID vid,
                                  size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_DIFF_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  auto x_view = getViewFromPointer(x, iend + 1);
  auto y_view = getViewFromPointer(y, iend + 1);

  auto firstdiff_lam = [=](Index_type i) { FIRST_DIFF_BODY; };

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      Kokkos::parallel_for(
          "FIRST_DIFF_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            x_view[i] = y_view[i + 1] - y_view[i];
          });
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  FIRST_DIFF : Unknown variant id = " << vid << std::endl;
  }
  }

  // View dimensions must match array dimensions!
  moveDataToHostFromKokkosView(x, x_view, iend + 1);
  moveDataToHostFromKokkosView(y, y_view, iend + 1);
}

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
