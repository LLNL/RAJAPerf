//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define NODAL_ACCUMULATION_3D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(x, m_x, m_nodal_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(vol, m_vol, m_zonal_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(real_zones, m_domain->real_zones, iend, did, hid);

#define NODAL_ACCUMULATION_3D_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_x, x, m_nodal_array_length, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(vol, did); \
  deallocOpenMPDeviceData(real_zones, did);


void NODAL_ACCUMULATION_3D::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_OMP_TARGET;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x0,x1,x2,x3,x4,x5,x6,x7, \
                                       vol, real_zones) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
        NODAL_ACCUMULATION_3D_BODY_INDEX;

        Real_type val = 0.125 * vol[i];

        #pragma omp atomic
        x0[i] += val;
        #pragma omp atomic
        x1[i] += val;
        #pragma omp atomic
        x2[i] += val;
        #pragma omp atomic
        x3[i] += val;
        #pragma omp atomic
        x4[i] += val;
        #pragma omp atomic
        x5[i] += val;
        #pragma omp atomic
        x6[i] += val;
        #pragma omp atomic
        x7[i] += val;
      }

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_OMP_TARGET;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    camp::resources::Resource working_res{camp::resources::Omp()};
    RAJA::TypedListSegment<Index_type> zones(m_domain->real_zones,
                                             m_domain->n_real_zones,
                                             working_res);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        zones, [=](Index_type i) {
        NODAL_ACCUMULATION_3D_RAJA_ATOMIC_BODY(RAJA::omp_atomic);
      });

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_OMP_TARGET;

  } else {
    getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
