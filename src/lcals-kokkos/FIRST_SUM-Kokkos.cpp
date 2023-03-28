//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_SUM.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void FIRST_SUM::runKokkosVariant(VariantID vid,
                                 size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize();

  FIRST_SUM_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  auto x_view = getViewFromPointer(x, iend);
  auto y_view = getViewFromPointer(y, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Kokkos::parallel_for(
          "FIRST_SUM_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            x_view[i] = y_view[i - 1] + y_view[i];
          });
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  FIRST_SUM : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(x, x_view, iend);
  moveDataToHostFromKokkosView(y, y_view, iend);
}

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
