//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

/*
void TRIAD::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();
*/

void TRIAD::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();


  TRIAD_DATA_SETUP;
/*
  #define TRIAD_DATA_SETUP \
  Real_ptr a = m_a; \
  Real_ptr b = m_b; \
  Real_ptr c = m_c; \
  Real_type alpha = m_alpha;
*/

  auto a_view = getViewFromPointer(a, iend);
  auto b_view = getViewFromPointer(b, iend);
  auto c_view = getViewFromPointer(c, iend);


  auto triad_lam = [=](Index_type i) {
                     TRIAD_BODY;
                   };

#if defined (RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIAD_BODY;
        }

      }
      stopTimer();

      break;
    }

// #if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          triad_lam(i);
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
          RAJA::RangeSegment(ibegin, iend), triad_lam);

      }
      stopTimer();

      break;
    }
    */

  case Kokkos_Lambda : {
                               Kokkos::fence();
                               startTimer();

                               for (RepIndex_type irep =0; irep < run_reps; ++irep) {
                                
                                Kokkos::parallel_for("TRIAD_Kokkos, Kokkos_Lambda",
                                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                       KOKKOS_LAMBDA(Index_type i) {
                                       // TRIAD_BODY definition in TRIAD.hpp
                                       //   a[i] = b[i] + alpha * c[i] ;
                                       a_view[i] = b_view[i] + alpha * c_view[i];
                                       });
                               }

                               Kokkos::fence();
                               stopTimer();

                               break;

                       }

//#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  TRIAD : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(a, a_view, iend);
  moveDataToHostFromKokkosView(b, b_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);

}

} // end namespace stream
} // end namespace rajaperf
