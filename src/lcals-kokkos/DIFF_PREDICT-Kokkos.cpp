//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void DIFF_PREDICT::runKokkosVariant(VariantID vid,
                                    size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DIFF_PREDICT_DATA_SETUP;

  // Wrapping pointers in Kokkos Views
  // Nota bene: get the actual array size to catch errors

  auto px_flat_view = getViewFromPointer(px, iend * 14);
  auto cx_flat_view = getViewFromPointer(cx, iend * 14);

  // 2D View w/ runtime and compile time dimension
  Kokkos::View<Real_type *[14], Kokkos::LayoutLeft> px_view(px_flat_view.data(),
                                                            iend);
  Kokkos::View<Real_type *[14], Kokkos::LayoutLeft> cx_view(cx_flat_view.data(),
                                                            iend);
  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Kokkos::parallel_for(
          "DIFF_PREDICT_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            // DIFF_PREDICT_BODY with Kokkos Views
            Real_type ar, br, cr;

            ar = cx_view(i, 4);
            br = ar - px_view(i, 4);
            px_view(i, 4) = ar;
            cr = br - px_view(i, 5);
            px_view(i, 5) = br;
            ar = cr - px_view(i, 6);
            px_view(i, 6) = cr;
            br = ar - px_view(i, 7);
            px_view(i, 7) = ar;
            cr = br - px_view(i, 8);
            px_view(i, 8) = br;
            ar = cr - px_view(i, 9);
            px_view(i, 9) = cr;
            br = ar - px_view(i, 10);
            px_view(i, 10) = ar;
            cr = br - px_view(i, 11);
            px_view(i, 11) = br;
            px_view(i, 13) = cr - px_view(i, 12);
            px_view(i, 12) = cr;
          });
    }
    Kokkos::fence();
    stopTimer();
    break;
  }

  default: {
    std::cout << "\n  DIFF_PREDICT : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(px, px_flat_view, iend * 14);
  moveDataToHostFromKokkosView(cx, cx_flat_view, iend * 14);
}

} // end namespace lcals
} // end namespace rajaperf
#endif
