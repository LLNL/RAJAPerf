//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {
void TRIDIAG_ELIM::runKokkosVariant(VariantID vid,
                                    size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  auto xout_view = getViewFromPointer(xout, iend);
  auto xin_view = getViewFromPointer(xin, iend);
  auto y_view = getViewFromPointer(y, iend);
  auto z_view = getViewFromPointer(z, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Kokkos::parallel_for(
          "TRIDIAG_ELIM_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            xout_view[i] = z_view[i] * (y_view[i] - xin_view[i - 1]);
          });
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  TRIDIAG_ELIM : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(xout, xout_view, iend);
  moveDataToHostFromKokkosView(xin, xin_view, iend);
  moveDataToHostFromKokkosView(y, y_view, iend);
  moveDataToHostFromKokkosView(z, z_view, iend);
}

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
