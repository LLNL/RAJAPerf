//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT3.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void INIT3::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();
  
  INIT3_DATA_SETUP;

  auto init3_lam = [=](Index_type i) {
                     INIT3_BODY;
                   };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {
       
      startTimer();
      for(RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for(Index_type i = ibegin; i < iend; ++i) {
	  INIT3_BODY;
        }

      }
      stopTimer();

      break;
}

#if defined(RUN_RAJA_SEQ)

    case Lambda_Seq : {
      
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        
        for (Index_type i = ibegin; i < iend; ++i) {
	  init3_lam(i);
        }


    }
    stopTimer();  
   
    break;
}

// Nota bene -- Conversion of Raja code begins here
    case Kokkos_Lambda : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//        RAJA::forall<RAJA::simd_exec>(
//          RAJA::RangeSegment(ibegin, iend), init3_lam);
          
         // Kokkos translation
        Kokkos::parallel_for("INIT3-KokkosSeq Kokkos_Lambda", Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend),
		[=] (Index_type i) {INIT3_BODY});
      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  INIT3 : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

}

} // end namespace basic
} // end namespace rajaperf
