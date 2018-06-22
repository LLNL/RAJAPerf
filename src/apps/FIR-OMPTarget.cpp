//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf 
{
namespace apps
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define FIR_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr in; \
  Real_ptr out; \
  Real_ptr coeff; \
\
  const Index_type coefflen = m_coefflen; \
\
  allocAndInitOpenMPDeviceData(in, m_in, getRunSize(), did, hid); \
  allocAndInitOpenMPDeviceData(out, m_out, getRunSize(), did, hid); \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocAndInitOpenMPDeviceData(coeff, tcoeff, FIR_COEFFLEN, did, hid);


#define FIR_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_out, out, getRunSize(), hid, did); \
  deallocOpenMPDeviceData(in, did); \
  deallocOpenMPDeviceData(out, did); \
  deallocOpenMPDeviceData(coeff, did);


void FIR::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  if ( vid == Base_OpenMPTarget ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(in, out, coeff) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
         FIR_BODY;
      }

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](int i) {
        FIR_BODY;
      });

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  FIR : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
