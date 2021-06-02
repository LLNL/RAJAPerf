//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

/*
void COPY::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();
*/

  void COPY::runKokkosVariant(VariantID vid)

  {
          const Index_type run_reps = getRunReps();
          const Index_type ibegin = 0;
          const Index_type iend = getRunSize();  


  COPY_DATA_SETUP;
        
  auto a_view = getViewFromPointer(a, iend);    
  auto c_view = getViewFromPointer(c, iend);


  auto copy_lam = [=](Index_type i) {
                    COPY_BODY;
                  };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          COPY_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          copy_lam(i);
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
          RAJA::RangeSegment(ibegin, iend), copy_lam);

      }
      stopTimer();

      break;
    }

    */

        case Kokkos_Lambda : {

                        
      Kokkos::fence();
      startTimer();     

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
              Kokkos::parallel_for("COPY_Kokkos Kokkos_Lambda",
              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin,iend),
              KOKKOS_LAMBDA(Index_type i) {
              // COPY BODY DEFINITION IN HEADER:
              //  c[i] = a[i] ;
              c_view[i] = a_view[i];
              });

      }          
      Kokkos::fence();
      stopTimer();      
                        
      break;            
    }                   



    default : {
      std::cout << "\n  COPY : Unknown variant id = " << vid << std::endl;
    }

  }


#endif //RUN_KOKKOS

  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);

}

} // end namespace stream
} // end namespace rajaperf
