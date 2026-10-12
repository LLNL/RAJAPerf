//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULADDSUB.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void MULADDSUB::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MULADDSUB_DATA_SETUP;

  // Define Kokkos Views that will wrap pointers defined in MULADDSUB.hpp
  auto out1_view = getViewFromPointer(out1, iend);
  auto out2_view = getViewFromPointer(out2, iend);
  auto out3_view = getViewFromPointer(out3, iend);
  auto in1_view = getViewFromPointer(in1, iend);
  auto in2_view = getViewFromPointer(in2, iend);

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();
    startTimer();

    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MULADDSUB_1");
      // If SIMD really matters , consider using Kokkos SIMD
      Kokkos::parallel_for(
          "MULTISUB-KokkosSeq Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type i) {
            out1_view[i] = in1_view[i] * in2_view[i];
            out2_view[i] = in1_view[i] + in2_view[i];
            out3_view[i] = in1_view[i] - in2_view[i];
          });
      RP_CALI_SUBKERNEL_END("MULADDSUB_1");
    }

    Kokkos::fence();
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  MULADDSUB : Unknown variant id = " << vid << std::endl;
  }
  }
  moveDataToHostFromKokkosView(out1, out1_view, iend);
  moveDataToHostFromKokkosView(out2, out2_view, iend);
  moveDataToHostFromKokkosView(out3, out3_view, iend);
  moveDataToHostFromKokkosView(out3, out3_view, iend);
  moveDataToHostFromKokkosView(in1, in1_view, iend);
  moveDataToHostFromKokkosView(in2, in2_view, iend);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MULADDSUB, Kokkos, Kokkos_Lambda)

} // end namespace basic
} // end namespace rajaperf
#endif
