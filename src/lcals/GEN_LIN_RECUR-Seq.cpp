//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{


void GEN_LIN_RECUR::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto genlinrecur_lam1 = [=](Index_type k) {
                            GEN_LIN_RECUR_BODY1;
                          };
  auto genlinrecur_lam2 = [=](Index_type i) {
                            GEN_LIN_RECUR_BODY2;
                          };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k ) {
          GEN_LIN_RECUR_BODY1;
        }

        for (Index_type i = 1; i < N+1; ++i ) {
          GEN_LIN_RECUR_BODY2;
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k ) {
          genlinrecur_lam1(k);
        }

        for (Index_type i = 1; i < N+1; ++i ) {
          genlinrecur_lam2(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(0, N), genlinrecur_lam1);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(1, N+1), genlinrecur_lam2);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace lcals
} // end namespace rajaperf
