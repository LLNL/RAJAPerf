//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void INIT_VIEW1D::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT_VIEW1D_DATA_SETUP;

  // Declare a Kokkos View that will be used to wrap a pointer 
  auto a_view = getViewFromPointer(a, iend);

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         Kokkos::parallel_for("INIT_VIEW1D_Kokkos Kokkos_Lambda",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin,iend),
                              KOKKOS_LAMBDA (Index_type i) {
                              //INIT_VIEW1D_BODY_RAJA
                              //Instead, use the INIT_VIEW1D_BODY definition
                              //with Kokkos View
                              //a[i] = (i+1) * v;
                              a_view[i] = (i + 1) * v;

                              });

      }

      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  INIT_VIEW1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(a, a_view, iend);

}

} // end namespace basic
} // end namespace rajaperf
