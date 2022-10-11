//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EOS.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void EOS::runKokkosVariant(VariantID vid,
                           size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  EOS_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  auto x_view = getViewFromPointer(x, iend + 7);
  auto y_view = getViewFromPointer(y, iend + 7);
  auto z_view = getViewFromPointer(z, iend + 7);
  auto u_view = getViewFromPointer(u, iend + 7);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      Kokkos::parallel_for(
          "EOS_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            x_view[i] =
                u_view[i] + r * (z_view[i] + r * y_view[i]) +
                t * (u_view[i + 3] + r * (u_view[i + 2] + r * u_view[i + 1]) +
                     t * (u_view[i + 6] +
                          q * (u_view[i + 5] + q * u_view[i + 4])));
          });
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  EOS : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(x, x_view, iend + 7);
  moveDataToHostFromKokkosView(y, y_view, iend + 7);
  moveDataToHostFromKokkosView(z, z_view, iend + 7);
  moveDataToHostFromKokkosView(u, u_view, iend + 7);
}

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
