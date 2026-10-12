//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void FIRST_DIFF::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_DIFF_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  auto x_view = getViewFromPointer(x, iend + 1);
  auto y_view = getViewFromPointer(y, iend + 1);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("FIRST_DIFF_1");
      Kokkos::parallel_for(
          "FIRST_DIFF_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            x_view[i] = y_view[i + 1] - y_view[i];
          });
      RP_CALI_SUBKERNEL_END("FIRST_DIFF_1");
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

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(FIRST_DIFF, Kokkos, Kokkos_Lambda)

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
