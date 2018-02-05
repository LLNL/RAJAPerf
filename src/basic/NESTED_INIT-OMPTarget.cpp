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

#include "NESTED_INIT.hpp"

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

#define NESTED_INIT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr array = m_array; \
  Index_type ni = m_ni; \
  Index_type nj = m_nj; \
  Index_type nk = m_nk; \
\
  allocAndInitOpenMPDeviceData(array, m_array, m_array_length, did, hid);

#define NESTED_INIT_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_array, array, m_array_length, hid, did); \
  deallocOpenMPDeviceData(array, did);


void NESTED_INIT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_OpenMPTarget ) {

    NESTED_INIT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(array) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(3) 
      for (Index_type k = 0; k < nk; ++k ) {
        for (Index_type j = 0; j < nj; ++j ) {
          for (Index_type i = 0; i < ni; ++i ) {
            NESTED_INIT_BODY;
          }
        }
      }  

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

#if 1 // temporary implementation until RAJA::nested::OmpCollapse works.

    NESTED_INIT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(0, nk), [=](Index_type k) {
        for (Index_type j = 0; j < nj; ++j ) {
          for (Index_type i = 0; i < ni; ++i ) {
            NESTED_INIT_BODY;
          }
        }
      });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_OMP_TARGET;

#else

    NESTED_INIT_DATA_SETUP_OMP_TARGET;

    using EXEC_POL = RAJA::nested::Policy<
                       RAJA::nested::OmpTargetCollapse<
                         RAJA::nested::For<2>,          //k
                         RAJA::nested::For<1>,          //j
                         RAJA::nested::For<0> > >;      //i

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::nested::forall(EXEC_POL{},
                           RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                            RAJA::RangeSegment(0, nj),
                                            RAJA::RangeSegment(0, nk)),
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_OMP_TARGET;

#endif                            

  } else { 
     std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
