//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP) \
 && _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define INDEXLIST_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
  \
  allocAndInitOpenMPDeviceData(x, m_x, iend, did, hid); \
  allocAndInitOpenMPDeviceData(list, m_list, iend, did, hid);

#define INDEXLIST_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_list, list, iend, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(list, did);

#endif


void INDEXLIST::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP) \
 && _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMPTarget : {

      INDEXLIST_DATA_SETUP_OMP_TARGET;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type count = 0;
        #pragma omp target is_device_ptr(x, list) device( did )
        #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) \
                                                  reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend; ++i ) {
          Index_type inc = 0;
          if (INDEXLIST_CONDITIONAL) {
            list[count] = i ;
            inc = 1;
          }
          #pragma omp scan exclusive(count)
          count += inc;
        }

        m_len = count;

      }
      stopTimer();

      INDEXLIST_DATA_TEARDOWN_OMP_TARGET;

      break;
    }

    default : {
      ignore_unused(run_reps, ibegin, iend, x, list);
      std::cout << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
