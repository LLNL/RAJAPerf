//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace basic {

void IF_QUAD::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  IF_QUAD_DATA_SETUP;

  // Instantiating views using getViewFromPointer for the IF_QUAD definition

  auto a_view = getViewFromPointer(a, iend);
  auto b_view = getViewFromPointer(b, iend);
  auto c_view = getViewFromPointer(c, iend);
  auto x1_view = getViewFromPointer(x1, iend);
  auto x2_view = getViewFromPointer(x2, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("IF_QUAD_1");
      Kokkos::parallel_for(
          "IF_QUAD_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            Real_type s = b_view[i] * b_view[i] - 4.0 * a_view[i] * c_view[i];
            if (s >= 0) {
              s = sqrt(s);
              x2_view[i] = (-b_view[i] + s) / (2.0 * a_view[i]);
              x1_view[i] = (-b_view[i] - s) / (2.0 * a_view[i]);
            } else {
              x2_view[i] = 0.0;
              x1_view[i] = 0.0;
            }
          });
      RP_CALI_SUBKERNEL_END("IF_QUAD_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  IF_QUAD : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(b, b_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);
  moveDataToHostFromKokkosView(x1, x1_view, iend);
  moveDataToHostFromKokkosView(x2, x2_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(IF_QUAD, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
