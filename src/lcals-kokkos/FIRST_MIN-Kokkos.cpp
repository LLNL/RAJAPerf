//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void FIRST_MIN::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_MIN_DATA_SETUP;

  // Wrap pointers in Kokkkos Views
  auto x_view = getViewFromPointer(x, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIRST_MIN_1");
      // The third template argument, `Kokkos::HostSpace`, is the memory space
      // where the result will be stored; the result will be stored in the same
      // place the kernel is called from , i.e., the Host
      using reducer_type =
          Kokkos::MinLoc<Real_type, Index_type, Kokkos::HostSpace>;
      // must hold the value and the location (host/device) ;
      // Create a custom-type variable to hold the result from parallel_reduce
      reducer_type::value_type min_result_obj;

      Kokkos::parallel_reduce(
          "FIRST_MIN_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i, reducer_type::value_type & mymin) {
            if (x_view[i] < mymin.val) {
              mymin.val = x_view[i];
              mymin.loc = i;
            }

            // Kokkos handles a MinLoc type
          },
          reducer_type(min_result_obj));

      m_minloc = min_result_obj.loc;
      RP_CALI_SUBKERNEL_END("FIRST_MIN_1");
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  FIRST_MIN : Unknown variant id = " << vid << std::endl;
  }
  }

  moveDataToHostFromKokkosView(x, x_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(FIRST_MIN, Kokkos, Kokkos_Lambda)

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
