//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define MAT_FUSED_MUL_ADD_DATA_SETUP_OMP_TARGET           \
  int hid = omp_get_initial_device();                     \
  int did = omp_get_default_device();                     \
  const Index_type N = m_N;                               \
  constexpr Index_type Ne = m_Ne;                         \
  const Index_type N_Elem = (N/(Ne*Ne);                   \
  allocAndInitOpenMPDeviceData(A, m_A, N, did, hid);      \
  allocAndInitOpenMPDeviceData(B, m_B, N, did, hid);      \
  allocAndInitOpenMPDeviceData(D, m_D, N, did, hid);			  

#define MAT_FUSED_MUL_ADD_DATA_TEARDOWN_OMP_TARGET        \
  getOpenMPDeviceData(m_A, A, N, hid, did);               \
  getOpenMPDeviceData(m_B, B, N, hid, did);               \
  getOpenMPDeviceData(m_D, D, N, hid, did);               \
  deallocOpenMPDeviceData(A, did);                        \
  deallocOpenMPDeviceData(B, did);                        \
  deallocOpenMPDeviceData(D, did);


void MAT_FUSED_MUL_ADD::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MAT_FUSED_MUL_ADD_DATA_SETUP;

  MAT_FUSED_MUL_ADD_DATA_INIT;

  if ( vid == Base_OpenMPTarget ) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      #pragma omp target is_device_ptr(A, B, D) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for(Index_type ii = 0; ii != N_Elem; ++ii){
        for(Index_type row = 0; row != Ne; ++row){
            for(Index_type col = 0; col != Ne; ++col){
            MAT_FUSED_MUL_ADD_BODY;
            }
          }
        }
      }
    
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_OMP_TARGET;

    RAJA::RangeSegment row_range(0, Ne);
    RAJA::RangeSegment col_range(0, Ne);    
    RAJA::RangeSegment ii_range(0, N_Elem);


    using EXEC_POL =
      RAJA::KernelPolicy<

        RAJA::statement::For<0, RAJA::seq_exec,         // ii
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<1, 2>, // row, col
            RAJA::statement::Lambda<0>
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(ii_range,
                                               row_range,
                                               col_range),
        [=] (Index_type ii, Index_type row, Index_type col) {
        MAT_FUSED_MUL_ADD_BODY;
      });
    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_OMP_TARGET;

  } else {
     getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown OMP Target variant id = " << vid << std::endl;
  }
}    
  }

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_TARGET_OPENMP
