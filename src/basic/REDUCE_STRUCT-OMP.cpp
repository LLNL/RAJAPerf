//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_STRUCT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <limits>
#include <iostream>

namespace rajaperf 
{
namespace basic
{

template < size_t tune_idx >
void REDUCE_STRUCT::runOpenMPVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("REDUCE_STRUCT_1");
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
        m_points = points;
        RP_CALI_SUBKERNEL_END("REDUCE_STRUCT_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto reduce_struct_x_base_lam = [=](Index_type i) -> Real_type {
                                   return x[i];
                                 };

      auto reduce_struct_y_base_lam = [=](Index_type i) -> Real_type {
                                   return y[i];
                                 };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("REDUCE_STRUCT_1");
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
        m_points = points;
        RP_CALI_SUBKERNEL_END("REDUCE_STRUCT_1");

      } 
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      if constexpr (tune_idx == 0) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE_STRUCT_1");
          RAJA::ReduceSum<RAJA::omp_reduce, Real_type> xsum(m_init_sum);
          RAJA::ReduceSum<RAJA::omp_reduce, Real_type> ysum(m_init_sum);
          RAJA::ReduceMin<RAJA::omp_reduce, Real_type> xmin(m_init_min); 
          RAJA::ReduceMin<RAJA::omp_reduce, Real_type> ymin(m_init_min);
          RAJA::ReduceMax<RAJA::omp_reduce, Real_type> xmax(m_init_max); 
          RAJA::ReduceMax<RAJA::omp_reduce, Real_type> ymax(m_init_max);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
              REDUCE_STRUCT_BODY_RAJA;
          });
  
          points.SetCenter((xsum.get()/(points.N)),
                           (ysum.get()/(points.N)));
          points.SetXMin((xmin.get())); 
          points.SetXMax((xmax.get()));
          points.SetYMin((ymin.get())); 
          points.SetYMax((ymax.get()));
          m_points = points;
          RP_CALI_SUBKERNEL_END("REDUCE_STRUCT_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE_STRUCT_1");
          Real_type txsum = m_init_sum;
          Real_type tysum = m_init_sum;
          Real_type txmin = m_init_min;
          Real_type tymin = m_init_min;
          Real_type txmax = m_init_max;
          Real_type tymax = m_init_max;

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&txsum),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tysum),
            RAJA::expt::Reduce<RAJA::operators::minimum>(&txmin),
            RAJA::expt::Reduce<RAJA::operators::minimum>(&tymin),
            RAJA::expt::Reduce<RAJA::operators::maximum>(&txmax),
            RAJA::expt::Reduce<RAJA::operators::maximum>(&tymax),
            [=](Index_type i, 
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& xsum,
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& ysum,
              RAJA::expt::ValOp<Real_type, RAJA::operators::minimum>& xmin,
              RAJA::expt::ValOp<Real_type, RAJA::operators::minimum>& ymin,
              RAJA::expt::ValOp<Real_type, RAJA::operators::maximum>& xmax,
              RAJA::expt::ValOp<Real_type, RAJA::operators::maximum>& ymax ) {
              REDUCE_STRUCT_BODY_RAJA;
            }
          );

          points.SetCenter(static_cast<Real_type>(txsum)/(points.N),
                           static_cast<Real_type>(tysum)/(points.N));
          points.SetXMin(static_cast<Real_type>(txmin));
          points.SetXMax(static_cast<Real_type>(txmax));
          points.SetYMin(static_cast<Real_type>(tymin));
          points.SetYMax(static_cast<Real_type>(tymax));
          m_points = points;
          RP_CALI_SUBKERNEL_END("REDUCE_STRUCT_1");

        }
        stopTimer();

      } else {
        getCout() << "\n  REDUCE_STRUCT : Unknown OpenMP tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  REDUCE_STRUCT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void REDUCE_STRUCT::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    addVariantTuning<&REDUCE_STRUCT::runOpenMPVariant<0>>(
        vid, "default");

    if (vid == RAJA_OpenMP) {

      addVariantTuning<&REDUCE_STRUCT::runOpenMPVariant<1>>(
          vid, "new");

    }

  }

}

} // end namespace basic
} // end namespace rajaperf

#endif
