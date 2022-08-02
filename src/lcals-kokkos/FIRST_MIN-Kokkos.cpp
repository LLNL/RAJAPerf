//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace lcals {

void FIRST_MIN::runKokkosVariant(VariantID vid,
                                 size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
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

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // The third template argument is the memory space where the
      // result will be stored; the result will be stored in the same place the
      // kernel is called from , i.e., the Host
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

} // end namespace lcals
} // end namespace rajaperf
#endif // RUN_KOKKOS
