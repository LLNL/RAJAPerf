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

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define LTIMES_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr phidat; \
  Real_ptr elldat; \
  Real_ptr psidat; \
\
  Index_type num_d = m_num_d; \
  Index_type num_z = m_num_z; \
  Index_type num_g = m_num_g; \
  Index_type num_m = m_num_m; \
\
  allocAndInitOpenMPDeviceData(phidat, m_phidat, m_philen, did, hid); \
  allocAndInitOpenMPDeviceData(elldat, m_elldat, m_elllen, did, hid); \
  allocAndInitOpenMPDeviceData(psidat, m_psidat, m_psilen, did, hid);

#define LTIMES_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_phidat, phidat, m_philen, hid, did); \
  deallocOpenMPDeviceData(phidat, did); \
  deallocOpenMPDeviceData(elldat, did); \
  deallocOpenMPDeviceData(psidat, did);


void LTIMES::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_OpenMPTarget ) {

    LTIMES_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(phidat, elldat, psidat) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(3)
      for (Index_type z = 0; z < num_z; ++z ) {
        for (Index_type g = 0; g < num_g; ++g ) {
          for (Index_type m = 0; m < num_m; ++m ) {
            for (Index_type d = 0; d < num_d; ++d ) {
              LTIMES_BODY;
            }
          }
        }
      }

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

#if 0 // disabled until RAJA::nested::OmpTargetCollapse works.

    LTIMES_DATA_SETUP_OMP_TARGET;

    LTIMES_VIEWS_RANGES_RAJA;

    using EXEC_POL = RAJA::nested::Policy<
                RAJA::nested::OmpTargetCollapse<
                   RAJA::nested::For<1>,                  // z
                   RAJA::nested::For<2>,                  // g
                   RAJA::nested::For<3> >,                // m
                 RAJA::nested::For<0, RAJA::loop_exec> >; // d

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::nested::forall(EXEC_POL{},
                           RAJA::make_tuple(IDRange(0, num_d),
                                            IZRange(0, num_z),
                                            IGRange(0, num_g),
                                            IMRange(0, num_m)),
        [=] (Index_type d, Index_type z, Index_type g, Index_type m) {
        LTIMES_BODY_RAJA;
      });

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_OMP_TARGET;

#endif

  } else {
     std::cout << "\n LTIMES : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
