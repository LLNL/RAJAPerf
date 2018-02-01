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

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define REDUCE3_INT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Int_ptr vec; \
\
  allocAndInitOpenMPDeviceData(vec, m_vec, iend, did, hid);

#define REDUCE3_INT_DATA_TEARDOWN_OMP_TARGET \
  deallocOpenMPDeviceData(vec, did); \


void REDUCE3_INT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    REDUCE3_INT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Int_type vsum = m_vsum_init;
      Int_type vmin = m_vmin_init;
      Int_type vmax = m_vmax_init;

      #pragma omp target is_device_ptr(vec) device( did ) map(tofrom:vsum, vmin, vmax)
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static,1) \
                               reduction(+:vsum) \
                               reduction(min:vmin) \
                               reduction(max:vmax)
      for (Index_type i = ibegin; i < iend; ++i ) {
        REDUCE3_INT_BODY;
      }

      m_vsum += vsum;
      m_vmin = RAJA_MIN(m_vmin, vmin);
      m_vmax = RAJA_MAX(m_vmax, vmax);

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    REDUCE3_INT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::omp_target_reduce<NUMTEAMS>, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::omp_target_reduce<NUMTEAMS>, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::omp_target_reduce<NUMTEAMS>, Int_type> vmax(m_vmax_init);

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend),
        [=](Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Real_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Real_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Real_type>(vmax.get()));

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  REDUCE3_INT : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
