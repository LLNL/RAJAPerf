//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace stream {

void COPY::runKokkosVariant(VariantID vid,
                            size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  COPY_DATA_SETUP;

  auto a_view = getViewFromPointer(a, iend);
  auto c_view = getViewFromPointer(c, iend);

  auto copy_lam = [=](Index_type i) { COPY_BODY; };

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      Kokkos::parallel_for(
          "COPY_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) { c_view[i] = a_view[i]; });
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  COPY : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);
}

} // end namespace stream
} // end namespace rajaperf
#endif // (RUN_KOKKOS)
