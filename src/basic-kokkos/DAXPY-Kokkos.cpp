//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>

namespace rajaperf {
namespace basic {

struct DaxpyFunctor {
  Real_ptr x;
  Real_ptr y;
  Real_type a;
  DaxpyFunctor(Real_ptr m_x, Real_ptr m_y, Real_type m_a)
      : x(m_x), y(m_y), a(m_a) {}
  void operator()(Index_type i) const { DAXPY_BODY; }
};

void DAXPY::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DAXPY_DATA_SETUP;

  auto x_view = getViewFromPointer(x, iend);
  auto y_view = getViewFromPointer(y, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("DAXPY_1");
      Kokkos::parallel_for(
          "DAXPY-Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) { y_view[i] += a * x_view[i]; });
      RP_CALI_SUBKERNEL_END("DAXPY_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }
  default: {
    std::cout << "\n  DAXPY : Unknown variant id = " << vid << std::endl;
  }
  }

  // Move data (i.e., pointer, KokkosView-wrapped ponter) back to the host from
  // the device

  moveDataToHostFromKokkosView(x, x_view, iend);
  moveDataToHostFromKokkosView(y, y_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DAXPY, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
