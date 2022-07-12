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
#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void REDUCE_STRUCT::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  REDUCE_STRUCT_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
        Real_type xsum = m_init_sum; Real_type ysum = m_init_sum;
        Real_type xmin = m_init_min; Real_type ymin = m_init_min;
        Real_type xmax = m_init_max; Real_type ymax = m_init_max;

        using Reduce_type =  std::array<Real_type,6>;
        Reduce_type result =
        std::transform_reduce( std::execution::par_unseq,
                               begin, end,
                               Reduce_type{ m_init_sum, m_init_min, m_init_max,   // x
                                            m_init_sum, m_init_min, m_init_max }, // y
                        [=](Reduce_type a, Reduce_type b) -> Reduce_type {
                             auto xsum = a[0] + b[0];
                             auto xmin  = std::min(a[1],b[1]);
                             auto xmax  = std::max(a[2],b[2]);
                             auto ysum = a[3] + b[3];
                             auto ymin  = std::min(a[4],b[4]);
                             auto ymax  = std::max(a[5],b[5]);
                             Reduce_type red{ xsum, xmin, xmax, ysum, ymin, ymax };
                             return red;
                        },
                        [=](Index_type i) -> Reduce_type {
                             Reduce_type val{ points.x[i], points.x[i], points.x[i],
                                              points.y[i], points.y[i], points.y[i] };
                             return val;

                        }
        );

        xsum = result[0];
        xmin = result[1];
        xmax = result[2];
        ysum = result[3];
        ymin = result[4];
        ymax = result[5];

        points.SetCenter(xsum/(points.N), ysum/(points.N));
        points.SetXMin(xmin); 
        points.SetXMax(xmax);
        points.SetYMin(ymin);
        points.SetYMax(ymax);
        m_points=points;

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

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

#warning needs parallel
        for (Index_type i = ibegin; i < iend; ++i ) {
          xsum += reduce_struct_x_base_lam(i);
          xmin = std::min(xmin, reduce_struct_x_base_lam(i));
          xmax = std::max(xmax, reduce_struct_x_base_lam(i));
          ysum += reduce_struct_y_base_lam(i);
          ymin = std::min(ymin, reduce_struct_y_base_lam(i));
          ymax = std::max(ymax, reduce_struct_y_base_lam(i));
        }

        points.SetCenter(xsum/(points.N), ysum/(points.N));
        points.SetXMin(xmin); 
        points.SetXMax(xmax);
        points.SetYMin(ymin); 
        points.SetYMax(ymax);
        m_points=points;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  REDUCE_STRUCT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
