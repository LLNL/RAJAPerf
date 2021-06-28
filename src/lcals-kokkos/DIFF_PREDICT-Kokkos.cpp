//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void DIFF_PREDICT::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();


  DIFF_PREDICT_DATA_SETUP;

  // Instiating KokkosViews using getViewFromPointer;
  // Wrapping pointers in KokkosViews

  // You need to know the actual array size here to catch errors;
  //
  auto px_view = getViewFromPointer(px, iend*14);
  auto cx_view = getViewFromPointer(cx, iend*14);
  
  // NOTA BENE:  in DIFF_PREDICT.hpp, this constant:
  // const Index_type offset = m_offset;

  auto diffpredict_lam = [=](Index_type i) {
                           DIFF_PREDICT_BODY;
                         };

  #if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          DIFF_PREDICT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          diffpredict_lam(i);
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
          RAJA::RangeSegment(ibegin, iend), diffpredict_lam);

      }
      stopTimer();

      break;
    }
*/

// Kokkos-ifying here:
//
    case Kokkos_Lambda : {

      // Define ar, br cr because you are not using the DIFF_PREDICT_BODY
      
      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
             
              Kokkos::parallel_for("DIFF_PREDICT_Kokkos Kokkos_Lambda",
/*   
(gdb) p offset
$1 = 100000
(gdb) 
$2 = 100000
(gdb) p iend
$3 = 100000
*/

                                   Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                   KOKKOS_LAMBDA(Index_type i) {
                                   // DIFF_PREDICT_BODY definition in
                                   // DIFF_PREDICT.hpp:
                                   /*
                                     ar                  = cx[i + offset * 4];       \
                                     br                  = ar - px[i + offset * 4];  \
                                     px[i + offset * 4]  = ar;                       \
                                     cr                  = br - px[i + offset * 5];  \
                                     px[i + offset * 5]  = br;                       \
                                     ar                  = cr - px[i + offset * 6];  \
                                     px[i + offset * 6]  = cr;                       \
                                     br                  = ar - px[i + offset * 7];  \
                                     px[i + offset * 7]  = ar;                       \
                                     cr                  = br - px[i + offset * 8];  \
                                     px[i + offset * 8]  = br;                       \
                                     ar                  = cr - px[i + offset * 9];  \
                                     px[i + offset * 9]  = cr;                       \
                                     br                  = ar - px[i + offset * 10]; \
                                     px[i + offset * 10] = ar;                       \
                                     cr                  = br - px[i + offset * 11]; \
                                     px[i + offset * 11] = br;                       \
                                     px[i + offset * 13] = cr - px[i + offset * 12]; \
                                     px[i + offset * 12] = cr;

                                     */

                                     Real_type ar, br, cr; 
                                     ar                  = cx_view[i + offset * 4];       \
                                     br                  = ar - px_view[i + offset * 4];  \
                                     px_view[i + offset * 4]  = ar;                       \
                                     cr                  = br - px_view[i + offset * 5];  \
                                     px_view[i + offset * 5]  = br;                       \
                                     ar                  = cr - px_view[i + offset * 6];  \
                                     px_view[i + offset * 6]  = cr;                       \
                                     br                  = ar - px_view[i + offset * 7];  \
                                     px_view[i + offset * 7]  = ar;                       \
                                     cr                  = br - px_view[i + offset * 8];  \
                                     px_view[i + offset * 8]  = br;                       \
                                     ar                  = cr - px_view[i + offset * 9];  \
                                     px_view[i + offset * 9]  = cr;                       \
                                     br                  = ar - px_view[i + offset * 10]; \
                                     px_view[i + offset * 10] = ar;                       \
                                     cr                  = br - px_view[i + offset * 11]; \
                                     px_view[i + offset * 11] = br;                       \
                                     px_view[i + offset * 13] = cr - px_view[i + offset * 12]; \
                                     px_view[i + offset * 12] = cr;
                                   });

      }
      Kokkos::fence();
      stopTimer();

      break;
    }


    default : {
      std::cout << "\n  DIFF_PREDICT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(px, px_view, iend*14);
  moveDataToHostFromKokkosView(cx, cx_view, iend*14);

}

} // end namespace lcals
} // end namespace rajaperf
