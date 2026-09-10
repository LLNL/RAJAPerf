//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
//
KOKKOS_FUNCTION
Real_type trap_int_func(Real_type x, Real_type y, Real_type xp, Real_type yp) {
  Real_type denom = (x - xp) * (x - xp) + (y - yp) * (y - yp);
  denom = 1.0 / sqrt(denom);
  return denom;
}

void TRAP_INT::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("TRAP_INT_1");
      Real_type trap_integral_val = m_sumx_init;

      Kokkos::parallel_reduce(
          "TRAP_INT_Kokkos Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(const int64_t i, Real_type &sumx){TRAP_INT_BODY},
          trap_integral_val);

      m_sumx += static_cast<Real_type>(trap_integral_val) * h;
      RP_CALI_SUBKERNEL_END("TRAP_INT_1");
    }
    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  TRAP_INT : Unknown variant id = " << vid << std::endl;
  }
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(TRAP_INT, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
