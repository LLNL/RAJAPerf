//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

// Delete me
// For de-bugging:
#include "RAJA/RAJA.hpp"

namespace rajaperf {
namespace basic {

void DAXPY_ATOMIC::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DAXPY_ATOMIC_DATA_SETUP;
  //
  // Kokkos Views to wrap pointers declared in DAXPY_ATOMIC.hpp
  //

  auto x_view = getViewFromPointer(x, iend);
  auto y_view = getViewFromPointer(y, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DAXPY_ATOMIC_1");
      Kokkos::parallel_for(
          "DAXPY_ATOMIC_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            Kokkos::atomic_add(&y_view[i], a * x_view[i]);
          });
      RP_CALI_SUBKERNEL_END("DAXPY_ATOMIC_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    getCout() << "\n  DAXPY_ATOMIC : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(x, x_view, iend);
  moveDataToHostFromKokkosView(y, y_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DAXPY_ATOMIC, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
