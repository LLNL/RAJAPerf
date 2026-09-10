//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void INIT_VIEW1D_OFFSET::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize() + 1;

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  auto a_view = getViewFromPointer(a, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
      Kokkos::parallel_for(
          "INIT_VIEW1D_OFFSET_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) { a_view[i - ibegin] = i * v; });
      RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown variant id = " << vid
              << std::endl;
  }
  }

  // Move data from Kokkos View (on Device) back to Host
  moveDataToHostFromKokkosView(a, a_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INIT_VIEW1D_OFFSET, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
