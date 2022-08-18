//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"
#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void NESTED_INIT::runKokkosVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  // Wrap the nested init array pointer in a Kokkos View
  // In  a Kokkos View, array arguments for array boundaries go from outmost
  // to innermost dimension sizes
  // See the basic NESTED_INIT.hpp file for defnition of NESTED_INIT

  auto array_kokkos_view = getViewFromPointer(array, nk, nj, ni);
  //
  // Used in Kokkos variant (below).  Do not remove.
  //
  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
    NESTED_INIT_BODY;
  };

  switch (vid) {

  case Kokkos_Lambda: {

    Kokkos::fence();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // MDRange can be optimized
      Kokkos::parallel_for(
          "NESTED_INIT KokkosSeq",
          // Range policy to define amount of work to be done
          Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                                // Execution space
                                Kokkos::DefaultExecutionSpace>({0, 0, 0},
                                                               {nk, nj, ni}),
          // Loop body
          KOKKOS_LAMBDA(Index_type k, Index_type j, Index_type i) {
            array_kokkos_view(k, j, i) = 0.00000001 * i * j * k;
          });
    }

    Kokkos::fence();

    stopTimer();
    // Moves mirror data from GPU to CPU (void, i.e., no return type).  In
    // this moving of data back to Host, the layout is changed back to Layout
    // Right, vs. the LayoutLeft of the GPU
    moveDataToHostFromKokkosView(array, array_kokkos_view, nk, nj, ni);

    break;
  }

  default: {
    std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
  }
  }
}

} // end namespace basic
} // end namespace rajaperf
#endif
