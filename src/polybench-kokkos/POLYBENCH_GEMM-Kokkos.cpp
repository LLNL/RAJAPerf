//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#if defined(RUN_KOKKOS)

#include "common/KokkosViewUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace polybench {

void POLYBENCH_GEMM::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  auto A_view = getViewFromPointer(A, ni, nk);
  auto B_view = getViewFromPointer(B, nk, nj);
  auto C_view = getViewFromPointer(C, ni, nj);

  switch (vid) {
    case Kokkos_Lambda: {

      Kokkos::fence();
      startTimer();

      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        Kokkos::parallel_for(
            "POLYBENCH_GEMM Kokkos_Lambda",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                                  Kokkos::IndexType<Index_type>>({0, 0},
                                                                 {ni, nj}),
            KOKKOS_LAMBDA(Index_type i, Index_type j) {
              Real_type dot = 0.0;
              C_view(i, j) *= beta;
              for (Index_type k = 0; k < nk; ++k) {
                dot += alpha * A_view(i, k) * B_view(k, j);
              }
              C_view(i, j) = dot;
            });
      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default: {
      std::cout << "\n  POLYBENCH_GEMM : Unknown variant id = " << vid
                << std::endl;
    }
  }

  moveDataToHostFromKokkosView(A, A_view, ni, nk);
  moveDataToHostFromKokkosView(B, B_view, nk, nj);
  moveDataToHostFromKokkosView(C, C_view, ni, nj);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GEMM, Kokkos, Kokkos_Lambda)

}  // end namespace polybench
}  // end namespace rajaperf
#endif  // RUN_KOKKOS
