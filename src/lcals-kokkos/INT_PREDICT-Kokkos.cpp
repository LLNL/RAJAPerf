//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void INT_PREDICT::runKokkosVariant(VariantID vid,
                                   size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INT_PREDICT_DATA_SETUP;

  // Wrap pointer in Kokkos View, and adjust indices
  auto px_view = getViewFromPointer(px, iend * 13);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Declare variables in INT_PREDICT.hpp
      Real_type dm22 = m_dm22;
      Real_type dm23 = m_dm23;
      Real_type dm24 = m_dm24;
      Real_type dm25 = m_dm25;
      Real_type dm26 = m_dm26;
      Real_type dm27 = m_dm27;
      Real_type dm28 = m_dm28;

      Kokkos::parallel_for(
          "INT_PREDICT_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            px_view[i] =
                dm28 * px_view[i + offset * 12] +
                dm27 * px_view[i + offset * 11] +
                dm26 * px_view[i + offset * 10] +
                dm25 * px_view[i + offset * 9] +
                dm24 * px_view[i + offset * 8] +
                dm23 * px_view[i + offset * 7] +
                dm22 * px_view[i + offset * 6] +
                c0 * (px_view[i + offset * 4] + px_view[i + offset * 5]) +
                px_view[i + offset * 2];
          });
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  INT_PREDICT : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(px, px_view, iend * 13);
}

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
