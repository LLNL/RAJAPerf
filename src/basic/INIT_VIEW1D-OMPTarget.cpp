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

#include "INIT_VIEW1D.hpp"

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

#define INIT_VIEW1D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitOpenMPDeviceData(a, m_a, iend, did, hid);

#define INIT_VIEW1D_DATA_SETUP_RAJA_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitOpenMPDeviceData(a, m_a, iend, did, hid); \
\
  using ViewType = RAJA::View<Real_type, RAJA::Layout<1> >; \
  const RAJA::Layout<1> my_layout(iend); \
  ViewType view(a, my_layout);

#define INIT_VIEW1D_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_a, a, iend, hid, did); \
  deallocOpenMPDeviceData(a, did);


void INIT_VIEW1D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    INIT_VIEW1D_DATA_SETUP_OMP_TARGET;                 

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(a) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        INIT_VIEW1D_BODY;
      }

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

     INIT_VIEW1D_DATA_SETUP_RAJA_OMP_TARGET

     startTimer();
     for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
         RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
         INIT_VIEW1D_BODY_RAJA;
       });

     }
     stopTimer();

     INIT_VIEW1D_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  INIT_VIEW1D : Unknown OMP Targetvariant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
