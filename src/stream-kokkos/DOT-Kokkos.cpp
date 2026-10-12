//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace stream {

void DOT::runKokkosVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  // Instantiation of pointer - wrapped Kokkos views:
  auto a_view = getViewFromPointer(a, iend);
  auto b_view = getViewFromPointer(b, iend);

  switch (vid) {

  case Kokkos_Lambda: {
    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DOT_1");
      Real_type dot = m_dot_init;

      parallel_reduce(
          "DOT-Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i, Real_type & dot_res) {
            dot_res += a_view[i] * b_view[i];
          },
          dot);
      m_dot += static_cast<Real_type>(dot);
      RP_CALI_SUBKERNEL_END("DOT_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  DOT : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(b, b_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DOT, Kokkos, Kokkos_Lambda)

} // end namespace stream
} // end namespace rajaperf
#endif // (RUN_KOKKOS)
