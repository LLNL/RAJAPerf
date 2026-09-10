//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"

#include <iostream>
#include <limits>
#include <algorithm>

namespace rajaperf {
namespace basic {

void REDUCE3_INT::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  // Declare KokkosView that will wrap the pointer to a vector

  auto vec_view = getViewFromPointer(vec, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("REDUCE3_INT_1");
      Int_type max_value = m_vmax_init;
      Int_type min_value = m_vmin_init;
      Int_type sum = m_vsum_init;
      // ADL: argument-dependent look up here
      parallel_reduce(
          "REDUCE3-Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(const int64_t i, Int_type &tl_max, Int_type &tl_min,
                        Int_type &tl_sum) {
            Int_type vec_i = vec_view[i];
            if (vec_i > tl_max)
              tl_max = vec_i;
            if (vec_i < tl_min)
              tl_min = vec_i;
            tl_sum += vec_i;
          },
          Kokkos::Max<Int_type>(max_value), Kokkos::Min<Int_type>(min_value),
          sum);
      m_vsum += static_cast<Int_type>(sum);
      m_vmin = std::min(m_vmin, static_cast<Int_type>(min_value));
      m_vmax = std::max(m_vmax, static_cast<Int_type>(max_value));
      RP_CALI_SUBKERNEL_END("REDUCE3_INT_1");
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  REDUCE3_INT : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(vec, vec_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(REDUCE3_INT, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
