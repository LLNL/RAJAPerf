//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace basic {

void PI_ATOMIC::runKokkosVariant(VariantID vid) {
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
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_ATOMIC_1");
      // Initializing a value, pi, on the host
      *pi = m_pi_init;

      pi_view = getViewFromPointer(pi, 1);

      Kokkos::parallel_for(
          "PI_ATOMIC-Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            Real_type x = (Real_type(i) + 0.5) * dx;
            // Make a reference to the 0th element of a 1D view with one
            // element
            Kokkos::atomic_add(&pi_view(0), dx / (1.0 + x * x));
          });
      // Moving the data on the device (held in the KokkosView) BACK to the
      // pointer, pi.
      moveDataToHostFromKokkosView(pi, pi_view, 1);
      *pi *= 4.0;
      RP_CALI_SUBKERNEL_END("PI_ATOMIC_1");
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

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PI_ATOMIC, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
