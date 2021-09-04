//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_STRUCT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define REDUCE_STRUCT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitHipDeviceData(particles.x, m_x, particles.N, did, hid); \
  allocAndInitHipDeviceData(particles.y, m_y, particles.N, did, hid); 

#define REDUCE_STRUCT_DATA_TEARDOWN_OMP_TARGET \
  deallocHipDeviceData(particles.x); \
  deallocHipDeviceData(particles.y); \


void REDUCE_STRUCT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    REDUCE_STRUCT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type xsum = 0.0;
      Real_type xmin = 0.0;
      Real_type xmax = 0.0;

      #pragma omp target is_device_ptr(vec) device( did ) map(tofrom:vsum, vmin, vmax)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static,1) \
                               reduction(+:xsum) \
                               reduction(min:xmin) \
                               reduction(max:xmax)
      for (Index_type i = ibegin; i < iend; ++i ) {
        REDUCE_STRUCT_BODY;
      }

      particles.SetCenter(xsum/particles.N,0.0);
      particles.SetXMin(xmin);
      particles.SetXMax(xmax);


    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    REDUCE_STRUCT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::omp_target_reduce, Real_type> xsum;
      RAJA::ReduceMin<RAJA::omp_target_reduce, Real_type> xmin;
      RAJA::ReduceMax<RAJA::omp_target_reduce, Real_type> xmax;

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend),
        [=](Index_type i) {
        REDUCE_STRUCT_BODY_RAJA;
      });

     particles.SetCenter(static_cast<Real_type>(xsum.get()/(particles.N+1)),0.0);
	 particles.SetXMin(static_cast<Real_type>(xmin.get()));
	 particles.SetXMax(static_cast<Real_type>(xmax.get()));

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  REDUCE_STRUCT : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
