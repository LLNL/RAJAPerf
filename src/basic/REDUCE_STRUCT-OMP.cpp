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

        Real_type xsum = 0.0; Real_type ysum = 0.0;
        Real_type xmin = 0.0; Real_type ymin = 0.0;
        Real_type xmax = 0.0; Real_type ymax = 0.0;

        #pragma omp parallel for reduction(+:xsum), \
                                 reduction(min:xmin), \
                                 reduction(max:xmax), \
                                 reduction(+:ysum), \
                                 reduction(min:ymin), \
                                 reduction(max:ymax)
        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE_STRUCT_BODY;
        }

        particles.SetCenter(xsum/particles.N,ysum/particles.N);
        particles.SetXMin(xmin); particles.SetXMax(xmax);
        particles.SetYMin(ymin); particles.SetYMax(ymax);
        m_particles=particles;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto reduce_struct_x_base_lam = [=](Index_type i) -> Real_type {
                                   return particles.x[i];
                                 };

      auto reduce_struct_y_base_lam = [=](Index_type i) -> Real_type {
                                   return particles.y[i];
                                 };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type xsum = 0.0, ysum = 0.0;
        Real_type xmin = 0.0, ymin = 0.0;
        Real_type xmax = 0.0, ymax = 0.0;

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

        particles.SetCenter(xsum/particles.N,ysum/particles.N);
        particles.SetXMin(xmin); particles.SetXMax(xmax);
        particles.SetYMin(ymin); particles.SetYMax(ymax);
        m_particles=particles;

      } 
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
        RAJA::ReduceSum<RAJA::omp_reduce, Real_type> xsum(0.0);
        RAJA::ReduceSum<RAJA::omp_reduce, Real_type> ysum(0.0);
        RAJA::ReduceMin<RAJA::omp_reduce, Real_type> xmin(0.0); 
        RAJA::ReduceMin<RAJA::omp_reduce, Real_type> ymin(0.0);
        RAJA::ReduceMax<RAJA::omp_reduce, Real_type> xmax(0.0); 
        RAJA::ReduceMax<RAJA::omp_reduce, Real_type> ymax(0.0);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          REDUCE_STRUCT_BODY_RAJA;
        });

        particles.SetCenter(static_cast<Real_type>(xsum.get()/(particles.N)),ysum.get()/(particles.N));
	    particles.SetXMin(static_cast<Real_type>(xmin.get())); particles.SetYMin(static_cast<Real_type>(xmax.get()));
	    particles.SetYMax(static_cast<Real_type>(ymax.get())); particles.SetYMax(static_cast<Real_type>(ymax.get()));
        m_particles=particles;

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  REDUCE_STRUCT : Unknown variant id = " << vid << std::endl;
    }

  }

#else 
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
