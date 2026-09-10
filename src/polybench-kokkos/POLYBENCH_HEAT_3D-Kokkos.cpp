//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#if defined(RUN_KOKKOS)

#include "common/KokkosViewUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace polybench {

void POLYBENCH_HEAT_3D::runKokkosVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  auto A_view = getViewFromPointer(A, N, N, N);
  auto B_view = getViewFromPointer(B, N, N, N);

  switch (vid) {
    case Kokkos_Lambda: {

      Kokkos::fence();
      startTimer();

      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_1");
        Kokkos::parallel_for(
            "POLYBENCH_HEAT_3D Kokkos_Lambda--BODY1",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                                  Kokkos::IndexType<Index_type>>(
                {1, 1, 1}, {N - 1, N - 1, N - 1}),
            KOKKOS_LAMBDA(Index_type i, Index_type j, Index_type k) {
              B_view(i, j, k) =
                  0.125 * (A_view(i + 1, j, k) - 2.0 * A_view(i, j, k) +
                           A_view(i - 1, j, k)) +
                  0.125 * (A_view(i, j + 1, k) - 2.0 * A_view(i, j, k) +
                           A_view(i, j - 1, k)) +
                  0.125 * (A_view(i, j, k + 1) - 2.0 * A_view(i, j, k) +
                           A_view(i, j, k - 1)) +
                  A_view(i, j, k);
            });
        RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_2");
        Kokkos::parallel_for(
            "POLYBENCH_HEAT_3D Kokkos_Lambda--BODY2",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                                  Kokkos::IndexType<Index_type>>(
                {1, 1, 1}, {N - 1, N - 1, N - 1}),
            KOKKOS_LAMBDA(Index_type i, Index_type j, Index_type k) {
              A_view(i, j, k) =
                  0.125 * (B_view(i + 1, j, k) - 2.0 * B_view(i, j, k) +
                           B_view(i - 1, j, k)) +
                  0.125 * (B_view(i, j + 1, k) - 2.0 * B_view(i, j, k) +
                           B_view(i, j - 1, k)) +
                  0.125 * (B_view(i, j, k + 1) - 2.0 * B_view(i, j, k) +
                           B_view(i, j, k - 1)) +
                  B_view(i, j, k);
            });
        RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_2");

      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default: {
      std::cout << "\n  POLYBENCH_HEAT_3D : Unknown variant id = " << vid
                << std::endl;
    }

    moveDataToHostFromKokkosView(A, A_view, N, N, N);
    moveDataToHostFromKokkosView(B, B_view, N, N, N);
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_HEAT_3D, Kokkos, Kokkos_Lambda)

}  // end namespace polybench
}  // end namespace rajaperf
#endif  // RUN_KOKKOS
