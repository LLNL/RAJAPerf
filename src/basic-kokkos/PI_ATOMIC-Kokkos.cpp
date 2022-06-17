//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace basic {

void PI_ATOMIC::runKokkosVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_ATOMIC_DATA_SETUP;

  // Declare Kokkos View that will wrap the pointer defined in PI_ATOMIC.hpp
  auto pi_view = getViewFromPointer(pi, 1);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Initializing a value, pi, on the host
      *pi = m_pi_init;

      pi_view = getViewFromPointer(pi, 1);

      Kokkos::parallel_for(
          "PI_ATOMIC-Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            double x = (double(i) + 0.5) * dx;
            // Make a reference to the 0th element of a 1D view with one
            // element
            Kokkos::atomic_add(&pi_view(0), dx / (1.0 + x * x));
          });
      // Moving the data on the device (held in the KokkosView) BACK to the
      // pointer, pi.
      moveDataToHostFromKokkosView(pi, pi_view, 1);
      *pi *= 4.0;
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  PI_ATOMIC : Unknown variant id = " << vid << std::endl;
  }
  }
}

} // end namespace basic
} // end namespace rajaperf
#endif
