//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#if defined(RUN_KOKKOS)

#include "common/KokkosViewUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace polybench {

void POLYBENCH_JACOBI_2D::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  auto A_view = getViewFromPointer(A, N, N);
  auto B_view = getViewFromPointer(B, N, N);

  switch (vid) {
    case Kokkos_Lambda: {

      Kokkos::fence();
      startTimer();

      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
        Kokkos::parallel_for(
            "JACOBI_2D_Kokkos Kokkos_Lambda--BODY1",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                                  Kokkos::IndexType<Index_type>>(
                {1, 1}, {N - 1, N - 1}),
            KOKKOS_LAMBDA(Index_type i, Index_type j) {
              B_view(i, j) =
                  0.2 * (A_view(i, j) + A_view(i, j - 1) + A_view(i, j + 1) +
                         A_view(i + 1, j) + A_view(i - 1, j));
            });
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
        Kokkos::parallel_for(
            "JACOBI_2D_Kokkos Kokkos_Lambda--BODY2",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                                  Kokkos::IndexType<Index_type>>(
                {1, 1}, {N - 1, N - 1}),
            KOKKOS_LAMBDA(Index_type i, Index_type j) {
              A_view(i, j) =
                  0.2 * (B_view(i, j) + B_view(i, j - 1) + B_view(i, j + 1) +
                         B_view(i + 1, j) + B_view(i - 1, j));
            });
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default: {
      std::cout << "\n  POLYBENCH_JACOBI_2D : Unknown variant id = " << vid
                << std::endl;
    }
  }

  moveDataToHostFromKokkosView(A, A_view, N, N);
  moveDataToHostFromKokkosView(B, B_view, N, N);
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_2D, Kokkos, Kokkos_Lambda)

}  // end namespace polybench
}  // end namespace rajaperf
#endif  // RUN_KOKKOS
