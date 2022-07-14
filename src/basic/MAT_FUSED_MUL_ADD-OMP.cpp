//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void MAT_FUSED_MUL_ADD::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();
  const Index_type N = m_N;
  const Index_type Ne = m_Ne;
  const Index_type N_Elem = (N/(Ne*Ne));
  MAT_FUSED_MUL_ADD_DATA_SETUP;

  MAT_FUSED_MUL_ADD_DATA_INIT;

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    #pragma omp parallel for
    for(Index_type ii = 0; ii < N_Elem; ++ii){
          for(Index_type row = 0; row < Ne; ++row){
            for(Index_type col = 0; col < Ne; ++col){
                MAT_FUSED_MUL_ADD_BODY;
            }
        }
    }


    }
    stopTimer();

    break;
  }

  case Lambda_OpenMP: {
    auto mat_fused_base_lam = [=](Index_type ii, Index_type row, Index_type col){
        MAT_FUSED_MUL_ADD_BODY;
        };

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    #pragma omp parallel for
    for(Index_type ii = 0; ii < N_Elem; ++ii){
          for(Index_type row = 0; row < Ne; ++row){
            for(Index_type col = 0; col < Ne; ++col){
                mat_fused_base_lam(ii, row, col);
            }
        }
    }


    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {
 
    RAJA::RangeSegment row_range(0, Ne);
    RAJA::RangeSegment col_range(0, Ne);    
    RAJA::RangeSegment ii_range(0, (N/(Ne*Ne)));

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forall<RAJA::omp_parallel_for_exec>( ii_range, [=](int ii) {
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

  default: {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown variant id = " << vid
              << std::endl;
  }
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
