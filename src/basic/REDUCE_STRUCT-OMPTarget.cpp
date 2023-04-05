//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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


void REDUCE_STRUCT::runOpenMPTargetVariant(VariantID vid, 
                                           size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    Real_ptr xa = points.x;
    Real_ptr ya = points.y;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type xsum = m_init_sum; Real_type ysum = m_init_sum;
      Real_type xmin = m_init_min; Real_type ymin = m_init_min;
      Real_type xmax = m_init_max; Real_type ymax = m_init_max;

      #pragma omp target is_device_ptr(xa, ya) device( did ) map(tofrom:xsum, xmin, xmax, ysum, ymin, ymax)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static,1) \
                               reduction(+:xsum) \
                               reduction(min:xmin) \
                               reduction(max:xmax), \
                               reduction(+:ysum), \
                               reduction(min:ymin), \
                               reduction(max:ymax)
      for (Index_type i = ibegin; i < iend; ++i ) {
        xsum += xa[i] ;
        xmin = RAJA_MIN(xmin, xa[i]) ;
        xmax = RAJA_MAX(xmax, xa[i]) ;
        ysum += ya[i] ;
        ymin = RAJA_MIN(ymin, ya[i]) ;
        ymax = RAJA_MAX(ymax, ya[i]) ;
      }

      points.SetCenter(xsum/points.N, ysum/points.N);
      points.SetXMin(xmin);
      points.SetXMax(xmax);
      points.SetYMin(ymin); 
      points.SetYMax(ymax);
      m_points=points;

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::omp_target_reduce, Real_type> xsum(m_init_sum);
      RAJA::ReduceSum<RAJA::omp_target_reduce, Real_type> ysum(m_init_sum);
      RAJA::ReduceMin<RAJA::omp_target_reduce, Real_type> xmin(m_init_min);
      RAJA::ReduceMin<RAJA::omp_target_reduce, Real_type> ymin(m_init_min);
      RAJA::ReduceMax<RAJA::omp_target_reduce, Real_type> xmax(m_init_max);
      RAJA::ReduceMax<RAJA::omp_target_reduce, Real_type> ymax(m_init_max);

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend),
        [=](Index_type i) {
        REDUCE_STRUCT_BODY_RAJA;
      });

      points.SetCenter(xsum.get()/(points.N),
                       ysum.get()/(points.N));
      points.SetXMin(xmin.get());
      points.SetXMax(xmax.get());
      points.SetYMin(ymin.get());
      points.SetYMax(ymax.get());
      m_points=points;

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
