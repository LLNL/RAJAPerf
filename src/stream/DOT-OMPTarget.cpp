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

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define DOT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr a; \
  Real_ptr b; \
\
  allocAndInitOpenMPDeviceData(a, m_a, iend, did, hid); \
  allocAndInitOpenMPDeviceData(b, m_b, iend, did, hid);

#define DOT_DATA_TEARDOWN_OMP_TARGET \
  deallocOpenMPDeviceData(a, did); \
  deallocOpenMPDeviceData(b, did);

void DOT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    DOT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type dot = m_dot_init;

      #pragma omp target is_device_ptr(a, b) device( did ) map(tofrom:dot)
      #pragma omp teams distribute parallel for reduction(+:dot) \
              num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        DOT_BODY;
      }

      m_dot += dot;

    }
    stopTimer();

    DOT_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    DOT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::omp_target_reduce<NUMTEAMS>, Real_type> dot(m_dot_init);

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        DOT_BODY;
      });

      m_dot += static_cast<Real_type>(dot.get());

    }
    stopTimer();

    DOT_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  DOT : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
