//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

// Kokkos-ification starts here:

void FIRST_DIFF::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_DIFF_DATA_SETUP;

// From FIRST_DIFF.hpp
/*
#define FIRST_DIFF_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y;

*/
// lcals = livermore compiler analysis loops suite
  // Instiating KokkosViews using getViewFromPointer;
  // Wrapping pointers in KokkosViews

// attn:  look at the definition in setup in FIRST_DIFF.cpp: 
	auto x_view = getViewFromPointer(x, iend + 1);
	auto y_view = getViewFromPointer(y, iend + 1);

    auto firstdiff_lam = [=](Index_type i) {
                         FIRST_DIFF_BODY;
                       };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_DIFF_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          firstdiff_lam(i);
        }

      }
      stopTimer();

      break;
    }

/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), firstdiff_lam);

      }

      stopTimer();

      break;
    }
*/

    // Kokkos-ifying here:
    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
              Kokkos::parallel_for("FIRST_DIFF_Kokkos Kokkos_Lambda",
                                   Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                   KOKKOS_LAMBDA(Index_type i) {
                                   /* #define FIRST_DIFF_BODY  \
                                    x[i] = y[i+1] - y[i];
                                    */
                                   x_view[i] = y_view[i + 1] - y_view[i]; 
                                   });

      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  FIRST_DIFF : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  // ATTN:  View dimensions must match array dimensions!
  moveDataToHostFromKokkosView(x, x_view, iend + 1);
  moveDataToHostFromKokkosView(y, y_view, iend + 1);


}

} // end namespace lcals
} // end namespace rajaperf
