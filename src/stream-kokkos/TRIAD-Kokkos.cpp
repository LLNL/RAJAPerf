//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace stream {

void TRIAD::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIAD_DATA_SETUP;

  auto a_view = getViewFromPointer(a, iend);
  auto b_view = getViewFromPointer(b, iend);
  auto c_view = getViewFromPointer(c, iend);

  switch (vid) {

  case Kokkos_Lambda: {
    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("TRIAD_1");
      Kokkos::parallel_for(
          "TRIAD_Kokkos, Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            a_view[i] = b_view[i] + alpha * c_view[i];
          });
      RP_CALI_SUBKERNEL_END("TRIAD_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  TRIAD : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(b, b_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(TRIAD, Kokkos, Kokkos_Lambda)

} // end namespace stream
} // end namespace rajaperf
#endif // (RUN_KOKKOS)
