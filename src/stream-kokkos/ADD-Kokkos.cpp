//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

// Start Kokkos-ifying here:
// Nota bene: the original RAJAPerf Suite code left for reference
 /*
void ADD::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();
*/

  void ADD::runKokkosVariant(VariantID vid)
  {
  const Index_type run_reps = getRunReps();                
  const Index_type ibegin = 0;                     
  const Index_type iend = getActualProblemSize();


  ADD_DATA_SETUP;

  // Instiating views using getViewFromPointer

  auto a_view = getViewFromPointer(a, iend);
  auto b_view = getViewFromPointer(b, iend);
  auto c_view = getViewFromPointer(c, iend);



  auto add_lam = [=](Index_type i) {
                   ADD_BODY;
                 };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ADD_BODY;
        }

      }
      stopTimer();

      break;
    }


    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          add_lam(i);
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
          RAJA::RangeSegment(ibegin, iend), add_lam);

      }
      stopTimer();

      break;
    }
*/

//////////////////////////////////////////////////////////////////////////////
// Kokkos -fying here:
//

    case Kokkos_Lambda : {

      // open Kokkos fence
      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
       Kokkos::parallel_for("ADD_Kokkos Kokkos_Lambda",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                             KOKKOS_LAMBDA(Index_type i){
                             // ADD BODY definition in header:
                             // c[i] = a[i] + b[i];
                              c_view[i] = a_view[i] + b_view[i];
                             });

      }
      // close Kokkos fence
      Kokkos::fence();
      stopTimer();

      break;
    }


    default : {
      std::cout << "\n  ADD : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS


  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(b, b_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);



}

} // end namespace stream
} // end namespace rajaperf
