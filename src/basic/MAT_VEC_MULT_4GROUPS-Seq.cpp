//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_VEC_MULT_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void MAT_VEC_MULT_4GROUPS::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MAT_VEC_MULT_4GROUPS_DATA_SETUP;

  auto mvm4g_lam = [=](Index_type i) {
                     MAT_VEC_MULT_4GROUPS_BODY;
                   };



  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        for (Index_type i = ibegin; i < iend; ++i ) {
            Real_ptr a = m_a + 16*i;
            Real_ptr x = m_x + 16*i;
            Real_ptr y = m_y + 16*i;
            for(Index_type j=0; j<4; j++){
	        for(Index_type l=0; l<4; l++){
		    y[j*4+l] = 0;
	        }
                for(Index_type k=0; k<4; k++){
                    for(Index_type l=0; l<4; l++){
                        y[j*4+l] += a[k*4+j] * x[4*k+l];
                    }
                }
            }

        }

      }
      stopTimer();


      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          mvm4g_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), mvm4g_lam);


      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MAT_VEC_MULT_4GROUPS : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
