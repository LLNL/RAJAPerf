//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void LTIMES::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

 auto phi = getViewFromPointer(phidat, num_z, num_g, num_m);
 auto psi = getViewFromPointer(psidat, num_z, num_g, num_d);
 auto ell = getViewFromPointer(elldat, num_m, num_d);
 
#if defined (RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

     // Kokkos uses MDRange to model tightly-nested loops
      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
	   Kokkos::parallel_for("LTIMES",
                             Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{num_z, num_g, num_m, num_d}),
                             KOKKOS_LAMBDA(int64_t z, int64_t g, int64_t m, int64_t d) {
							 // #define LTIMES_BODY_RAJA \
                             // phi(z, g, m) +=  ell(m, d) * psi(z, g, d);
                             // make view named phi from phi dat
                              phi(z, g, m) +=  ell(m, d) * psi(z, g, d);


}); 



      }
      Kokkos::fence();
      stopTimer(); 

      break;
    }

    default : {
      std::cout << "\n LTIMES : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS 

  moveDataToHostFromKokkosView(phidat, phi, num_z, num_g, num_m);
  moveDataToHostFromKokkosView(psidat, psi, num_z, num_g, num_d);
  moveDataToHostFromKokkosView(elldat, ell, num_m, num_d);

}

} // end namespace apps
} // end namespace rajaperf
