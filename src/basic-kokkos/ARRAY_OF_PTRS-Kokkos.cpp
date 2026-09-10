//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ARRAY_OF_PTRS.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace basic {

void ARRAY_OF_PTRS::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ARRAY_OF_PTRS_DATA_SETUP;

  using view_type = std::decay_t<decltype(getViewFromPointer(x[0], iend))>;
  view_type x_view[ARRAY_OF_PTRS_MAX_ARRAY_SIZE];
  for (Index_type a = 0; a < array_size; ++a) {
    x_view[a] = getViewFromPointer(x[a], iend) ;
  }
  auto y_view = getViewFromPointer(y, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("ARRAY_OF_PTRS_1");
      Kokkos::parallel_for(
          "ARRAY_OF_PTRS-Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            y_view[i] = 0.0;
            for (Index_type a = 0; a < array_size; ++a) {
              y_view[i] += a * x_view[a][i];
            }
          });
      RP_CALI_SUBKERNEL_END("ARRAY_OF_PTRS_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }
  default: {
    std::cout << "\n  ARRAY_OF_PTRS : Unknown variant id = " << vid << std::endl;
  }
  }

  // Move data (i.e., pointer, KokkosView-wrapped ponter) back to the host from
  // the device

  moveDataToHostFromKokkosView(y, y_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(ARRAY_OF_PTRS, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
