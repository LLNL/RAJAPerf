//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void HYDRO_1D::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HYDRO_1D_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  auto x_view = getViewFromPointer(x, iend + 12);
  auto y_view = getViewFromPointer(y, iend + 12);
  auto z_view = getViewFromPointer(z, iend + 12);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_1D_1");
      Kokkos::parallel_for(
          "HYDRO_1D_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            x_view[i] =
                q + y_view[i] * (r * z_view[i + 10] + t * z_view[i + 11]);
          });
      RP_CALI_SUBKERNEL_END("HYDRO_1D_1");
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

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HYDRO_1D, Kokkos, Kokkos_Lambda)

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
