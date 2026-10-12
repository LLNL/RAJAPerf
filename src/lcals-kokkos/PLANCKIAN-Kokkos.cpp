//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <cmath>
#include <iostream>

namespace rajaperf {
namespace lcals {

void PLANCKIAN::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PLANCKIAN_DATA_SETUP;

  auto x_view = getViewFromPointer(x, iend);
  auto y_view = getViewFromPointer(y, iend);
  auto u_view = getViewFromPointer(u, iend);
  auto v_view = getViewFromPointer(v, iend);
  auto w_view = getViewFromPointer(w, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PLANCKIAN_1");
      Kokkos::parallel_for(
          "PLANCKIAN_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            y_view[i] = u_view[i] / v_view[i];
            w_view[i] = x_view[i] / (exp(y_view[i]) - 1.0);
          });
      RP_CALI_SUBKERNEL_END("PLANCKIAN_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  PLANCKIAN : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(x, x_view, iend);
  moveDataToHostFromKokkosView(y, y_view, iend);
  moveDataToHostFromKokkosView(u, u_view, iend);
  moveDataToHostFromKokkosView(v, v_view, iend);
  moveDataToHostFromKokkosView(w, w_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PLANCKIAN, Kokkos, Kokkos_Lambda)

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
