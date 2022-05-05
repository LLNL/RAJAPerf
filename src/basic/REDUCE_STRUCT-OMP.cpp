//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_STRUCT.hpp"

#include "RAJA/RAJA.hpp"

#include <limits>
#include <iostream>

namespace rajaperf 
{
namespace basic
{


void REDUCE_STRUCT::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type xsum = m_init_sum; Real_type ysum = m_init_sum;
        Real_type xmin = m_init_min; Real_type ymin = m_init_min;
        Real_type xmax = m_init_max; Real_type ymax = m_init_max;

        #pragma omp parallel for reduction(+:xsum), \
                                 reduction(min:xmin), \
                                 reduction(max:xmax), \
                                 reduction(+:ysum), \
                                 reduction(min:ymin), \
                                 reduction(max:ymax)
        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE_STRUCT_BODY;
        }

        points.SetCenter(xsum/points.N, ysum/points.N);
        points.SetXMin(xmin); 
        points.SetXMax(xmax);
        points.SetYMin(ymin); 
        points.SetYMax(ymax);
        m_points=points;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto reduce_struct_x_base_lam = [=](Index_type i) -> Real_type {
                                   return points.x[i];
                                 };

      auto reduce_struct_y_base_lam = [=](Index_type i) -> Real_type {
                                   return points.y[i];
                                 };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type xsum = m_init_sum; Real_type ysum = m_init_sum;
        Real_type xmin = m_init_min; Real_type ymin = m_init_min;
        Real_type xmax = m_init_max; Real_type ymax = m_init_max;

        #pragma omp parallel for reduction(+:xsum), \
                                 reduction(min:xmin), \
                                 reduction(max:xmax), \
                                 reduction(+:ysum), \
                                 reduction(min:ymin), \
                                 reduction(max:ymax)
        for (Index_type i = ibegin; i < iend; ++i ) {
          xsum += reduce_struct_x_base_lam(i);
          xmin = RAJA_MIN(xmin, reduce_struct_x_base_lam(i));
          xmax = RAJA_MAX(xmax, reduce_struct_x_base_lam(i));
          ysum += reduce_struct_y_base_lam(i);
          ymin = RAJA_MIN(ymin, reduce_struct_y_base_lam(i));
          ymax = RAJA_MAX(ymax, reduce_struct_y_base_lam(i));
        }

        points.SetCenter(xsum/points.N, ysum/points.N);
        points.SetXMin(xmin); 
        points.SetXMax(xmax);
        points.SetYMin(ymin);
        points.SetYMax(ymax);
        m_points=points;

      } 
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
        RAJA::ReduceSum<RAJA::omp_reduce, Real_type> xsum(m_init_sum);
        RAJA::ReduceSum<RAJA::omp_reduce, Real_type> ysum(m_init_sum);
        RAJA::ReduceMin<RAJA::omp_reduce, Real_type> xmin(m_init_min); 
        RAJA::ReduceMin<RAJA::omp_reduce, Real_type> ymin(m_init_min);
        RAJA::ReduceMax<RAJA::omp_reduce, Real_type> xmax(m_init_max); 
        RAJA::ReduceMax<RAJA::omp_reduce, Real_type> ymax(m_init_max);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          REDUCE_STRUCT_BODY_RAJA;
        });

        points.SetCenter((xsum.get()/(points.N)),
                         (ysum.get()/(points.N)));
        points.SetXMin((xmin.get())); 
        points.SetXMax((xmax.get()));
        points.SetYMin((ymin.get())); 
        points.SetYMax((ymax.get()));
        m_points=points;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  REDUCE_STRUCT : Unknown variant id = " << vid << std::endl;
    }

  }

#else 
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
