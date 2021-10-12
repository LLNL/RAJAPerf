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

  FIR_DATA_SETUP;

  // Wrap 4x4 array, "coeff" in a Kokkos::View;
  // "coeff" is used in the FIR_BODY
  // Real_type coeff[FIR_COEFFLEN];
  // Macro for 4x4 input array
   FIR_COEFF;
   // "coeff" is assined the memory location containing the value of the 0th element of coeff_array;
   Real_ptr coeff = &coeff_array[0];

  auto coeff_view = getViewFromPointer(coeff, FIR_COEFFLEN);

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
                                    sum += coeff_view[j]*in_view[i+j];
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
