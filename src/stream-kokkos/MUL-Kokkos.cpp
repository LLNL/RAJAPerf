//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MUL.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace stream {

void MUL::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MUL_DATA_SETUP;

  auto b_view = getViewFromPointer(b, iend);
  auto c_view = getViewFromPointer(c, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MUL_1");
      Kokkos::parallel_for(
          "MUL_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) { b_view[i] = alpha * c_view[i]; });
      RP_CALI_SUBKERNEL_END("MUL_1");
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  MUL : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(b, b_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MUL, Kokkos, Kokkos_Lambda)

} // end namespace stream
} // end namespace rajaperf
#endif // (RUN_KOKKOS)
