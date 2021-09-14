//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf 
{
namespace apps
{


void FIR::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize() - m_coefflen;

  // Macro for 1D Array of defined length of coefficients
  FIR_COEFF;

  // Declare & initialize pointers, coefflen
  FIR_DATA_SETUP;

  // Declare coeff array
  Real_type coeff[FIR_COEFFLEN];


  //  std::copy(iterator source_first, iterator source_end, iterator target_start);
  // Copy the "coeff_array" (in FIR.hpp) into the "coeff" array; both are
  // "Real_type" 
  std::copy(std::begin(coeff_array), std::end(coeff_array), std::begin(coeff));

  auto in_view =  getViewFromPointer(in, iend +  m_coefflen);
  auto out_view = getViewFromPointer(out, iend + m_coefflen);

  auto fir_lam = [=](Index_type i) {
                   FIR_BODY;
                 };
  
#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          FIR_BODY;
        }

      }
      stopTimer();

      break;
    } 

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
           fir_lam(i);
        }

      }
      stopTimer();

      break;
    }
/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), fir_lam);

      }
      stopTimer(); 

      break;
    }

    */


    case Kokkos_Lambda : {
      
      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("FIR - Kokkos_Lambda",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                              KOKKOS_LAMBDA(Index_type i) {
                              // #define FIR_BODY
                                  Real_type sum = 0.0;

                                  for (Index_type j = 0; j < coefflen; ++j ) {
                                    sum += coeff[j]*in_view[i+j];
                                  } 
                                  out_view[i] = sum;
                                  });

      }
      Kokkos::fence();
      stopTimer(); 

      break;
    }

    default : {
      std::cout << "\n  FIR : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS
  
  moveDataToHostFromKokkosView(in, in_view, iend + m_coefflen);
  moveDataToHostFromKokkosView(out, out_view, iend + m_coefflen);


}

} // end namespace apps
} // end namespace rajaperf
