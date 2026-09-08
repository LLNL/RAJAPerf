//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void DIFF_PREDICT::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DIFF_PREDICT_DATA_SETUP;

  // DIFF_PREDICT_DATA_SETUP shifts both pointers back by offset * 4, so
  // px + offset * 4 is m_px (10 * iend elements) and cx + offset * 4 is
  // m_cx (iend elements)

  auto px_flat_view = getViewFromPointer(px + offset * 4, iend * 10);
  auto cx_view = getViewFromPointer(cx + offset * 4, iend);

  // 2D View w/ runtime and compile time dimension
  Kokkos::View<Real_type *[10], Kokkos::LayoutLeft> px_view(px_flat_view.data(),
                                                            iend);
  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DIFF_PREDICT_1");
      Kokkos::parallel_for(
          "DIFF_PREDICT_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            // DIFF_PREDICT_BODY with Kokkos Views
            Real_type ar, br, cr;

            ar = cx_view(i);
            br = ar - px_view(i, 0);
            px_view(i, 0) = ar;
            cr = br - px_view(i, 1);
            px_view(i, 1) = br;
            ar = cr - px_view(i, 2);
            px_view(i, 2) = cr;
            br = ar - px_view(i, 3);
            px_view(i, 3) = ar;
            cr = br - px_view(i, 4);
            px_view(i, 4) = br;
            ar = cr - px_view(i, 5);
            px_view(i, 5) = cr;
            br = ar - px_view(i, 6);
            px_view(i, 6) = ar;
            cr = br - px_view(i, 7);
            px_view(i, 7) = br;
            px_view(i, 9) = cr - px_view(i, 8);
            px_view(i, 8) = cr;
          });
      RP_CALI_SUBKERNEL_END("DIFF_PREDICT_1");
    }
    Kokkos::fence();
    stopTimer();
    break;
  }

  default: {
    std::cout << "\n  DIFF_PREDICT : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(px + offset * 4, px_flat_view, iend * 10);
  moveDataToHostFromKokkosView(cx + offset * 4, cx_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DIFF_PREDICT, Kokkos, Kokkos_Lambda)

} // end namespace lcals
} // end namespace rajaperf
#endif
