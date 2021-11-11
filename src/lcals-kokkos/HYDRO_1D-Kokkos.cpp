//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void HYDRO_1D::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HYDRO_1D_DATA_SETUP;

  // Wrap pointers in Kokkos Views
  /*
   * #define HYDRO_1D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z;
  */

  auto x_view = getViewFromPointer(x, iend + 12);
  auto y_view = getViewFromPointer(y, iend + 12);
  auto z_view = getViewFromPointer(z, iend + 12);


  auto hydro1d_lam = [=](Index_type i) {
                       HYDRO_1D_BODY;
                     };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          HYDRO_1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          hydro1d_lam(i);
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
          RAJA::RangeSegment(ibegin, iend), hydro1d_lam);

      }
      stopTimer();

      break;
    }

    */



    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("HYDRO_1D_Kokkos Kokkos_Lambda",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                             // #define HYDRO_1D_BODY                                                          
                             // x[i] = q + y[i]*( r*z[i+10] + t*z[i+11] );
                             KOKKOS_LAMBDA(Index_type i) {
                             x_view[i] = q + y_view[i]*( r*z_view[i+10] + t*z_view[i+11] );
                             });

      }

      Kokkos::fence();
      stopTimer();

      break;
    }


    default : {
      std::cout << "\n  HYDRO_1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

    // ATTN:  Adjust arr dimensions to be congruent with the setup 
    // in the .cpp file:
    // m_array_length = getActualProblemSize() + 12;


    moveDataToHostFromKokkosView(x, x_view, iend + 12);
    moveDataToHostFromKokkosView(y, y_view, iend + 12);
    moveDataToHostFromKokkosView(z, z_view, iend + 12);

}

} // end namespace lcals
} // end namespace rajaperf
