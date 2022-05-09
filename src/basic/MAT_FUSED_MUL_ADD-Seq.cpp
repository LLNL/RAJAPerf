//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void MAT_FUSED_MUL_ADD::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  constexpr Index_type Ne = m_Ne;
  constexpr Index_type NeNe = m_Ne * m_Ne;

  MAT_FUSED_MUL_ADD_DATA_SETUP;

  MAT_FUSED_MUL_ADD_DATA_INIT;
  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){
              for(Index_type row = 0; row != Ne; ++row){
                for(Index_type col = 0; col != Ne; ++col){
                    MAT_FUSED_MUL_ADD_BODY;
                }
            }
        }

    } // number of iterations
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case Lambda_Seq: {

    auto mat_fused_lam = [=](Index_type ii, Index_type row, Index_type col){
        MAT_FUSED_MUL_ADD_BODY;
        };

    startTimer();
    for (Index_type irep = 0; irep < run_reps; ++irep) {
        for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){
              for(Index_type row = 0; row != Ne; ++row){
                for(Index_type col = 0; col != Ne; ++col){
                    mat_fused_lam(ii,row,col);
                    }
                 }
         }
                    
    } // irep
    stopTimer();

    break;
  }

  case RAJA_Seq: {
    RAJA::RangeSegment row_range(0, Ne);
    RAJA::RangeSegment col_range(0, Ne);    
    RAJA::RangeSegment ii_range(0, (N/(Ne*Ne)));

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forall<RAJA::loop_exec>( ii_range, [=](int ii) {
            RAJA::forall<RAJA::loop_exec>( row_range, [=](int row) {
                RAJA::forall<RAJA::loop_exec>( col_range, [=](int col) {    
                    MAT_FUSED_MUL_ADD_BODY;
                });
            });
         });
    }  // loop over kernel reps
    stopTimer();

    break;
  }
#endif // RUN_RAJA_SEQ

  default: {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown variant id = " << vid
              << std::endl;
  }
  }
}

} // end namespace basic
} // end namespace rajaperf
