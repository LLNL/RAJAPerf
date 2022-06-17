//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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

void DAXPY_ATOMIC::runKokkosVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
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

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Kokkos::parallel_for(
          "DAXPY_ATOMIC_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            Kokkos::atomic_add(&y_view[i], a * x_view[i]);
          });
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

} // end namespace basic
} // end namespace rajaperf
#endif
