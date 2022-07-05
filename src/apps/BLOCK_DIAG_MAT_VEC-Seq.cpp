//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "BLOCK_DIAG_MAT_VEC.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

void BLOCK_DIAG_MAT_VEC::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  constexpr Index_type ndof = m_ndof;

  BLOCK_DIAG_MAT_VEC_DATA_SETUP;
  
  BLOCK_DIAG_MAT_VEC_DATA_INIT;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  		for(Index_type e=0; e<N; ++e) {                                                                                                                                                             
  		    for (Index_type c=0; c<ndof; ++c)                                                                                                                                                               
  		    {                                                                                                                                                                                          
				BLOCK_DIAG_MAT_VEC_BODY;
  		    }                                                                                                                                                                                          
  		 }
    } // number of iterations
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case Lambda_Seq: {

    auto mat_vec_lam = [=](Index_type e, Index_type c){
        BLOCK_DIAG_MAT_VEC_BODY;
        };

    startTimer();
    for (Index_type irep = 0; irep < run_reps; ++irep) {
  		for(Index_type e=0; e<N; ++e) {                                                                                                                                                             
  		    for (Index_type c=0; c<ndof; ++c)                                                                                                                                                               
  		    {                                                                                                                                                                                          
				mat_vec_lam(e,c);
  		    }                                                                                                                                                                                          
  		 }
    } // irep
    stopTimer();

    break;
  }

  case RAJA_Seq: {
    RAJA::RangeSegment c_range(0, ndof);
    RAJA::RangeSegment e_range(0, N);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
		RAJA::forall<RAJA::loop_exec>( e_range, [=](int e) {
			RAJA::forall<RAJA::loop_exec>( c_range, [=](int c) {
				BLOCK_DIAG_MAT_VEC_BODY;
			});	
		});
    }  // loop over kernel reps
    stopTimer();

    break;
  }
#endif // RUN_RAJA_SEQ

  default: {
    getCout() << "\n  BLOCK_DIAG_MAT_VEC : Unknown variant id = " << vid
              << std::endl;
  }
  }
}

} // end namespace apps
} // end namespace rajaperf
