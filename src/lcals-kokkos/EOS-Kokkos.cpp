//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EOS.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void EOS::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  EOS_DATA_SETUP;

  auto x_view = getViewFromPointer(x, iend + 7);
  auto y_view = getViewFromPointer(y, iend + 7);
  auto z_view = getViewFromPointer(z, iend + 7);
  auto u_view = getViewFromPointer(u, iend + 7);


  auto eos_lam = [=](Index_type i) {
                   EOS_BODY;
                 };

  
#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          EOS_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          eos_lam(i);
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
          RAJA::RangeSegment(ibegin, iend), eos_lam);

      }
      stopTimer();

      break;
    }

*/

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
              Kokkos::parallel_for("EOS_Kokkos Kokkos_Lambda",
                                   Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                   KOKKOS_LAMBDA(Index_type i) {
                                   /*
								   #define EOS_BODY  \    
								   x[i] = u[i] + r*( z[i] + r*y[i] ) + \
                                       t*( u[i+3] + r*( u[i+2] + r*u[i+1] ) + \
                                           t*( u[i+6] + q*( u[i+5] + q*u[i+4] ) ) ); 
				                   */
                                   // Declare variables need in the function
                                   // body
                                   //const Real_type q;
                                   //const Real_type r;
                                   //const Real_type t;

								   x_view[i] = u_view[i] + r*( z_view[i] + r*y_view[i] ) + \
                                       t*( u_view[i+3] + r*( u_view[i+2] + r*u_view[i+1] ) + \
                                           t*( u_view[i+6] + q*( u_view[i+5] + q*u_view[i+4] ) ) ); 
                                   });

      }
      Kokkos::fence();
      stopTimer();

      break;
    }



    default : {
      std::cout << "\n  EOS : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(x, x_view, iend + 7);
  moveDataToHostFromKokkosView(y, y_view, iend + 7);
  moveDataToHostFromKokkosView(z, z_view, iend + 7);
  moveDataToHostFromKokkosView(u, u_view, iend + 7);


}

} // end namespace lcals
} // end namespace rajaperf
