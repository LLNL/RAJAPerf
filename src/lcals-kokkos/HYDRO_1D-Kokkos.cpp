//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void HYDRO_1D::runKokkosVariant(VariantID vid,
                                size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HYDRO_1D_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  auto x_view = getViewFromPointer(x, iend + 12);
  auto y_view = getViewFromPointer(y, iend + 12);
  auto z_view = getViewFromPointer(z, iend + 12);

  auto hydro1d_lam = [=](Index_type i) { HYDRO_1D_BODY; };

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Kokkos::parallel_for(
          "HYDRO_1D_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            x_view[i] =
                q + y_view[i] * (r * z_view[i + 10] + t * z_view[i + 11]);
          });
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  HYDRO_1D : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(x, x_view, iend + 12);
  moveDataToHostFromKokkosView(y, y_view, iend + 12);
  moveDataToHostFromKokkosView(z, z_view, iend + 12);
}

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
