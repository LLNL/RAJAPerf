//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include <array>
#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void REDUCE3_INT::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  REDUCE3_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        typedef std::array<Int_type,3> Reduce_type;
        Reduce_type result =
        std::transform_reduce( std::execution::par_unseq,
                               begin, end,
                               Reduce_type{m_vsum_init,m_vmin_init,m_vmax_init},
                        [=](Reduce_type a, Reduce_type b) -> Reduce_type {
                             auto plus = a[0] + b[0];
                             auto min  = std::min(a[1],b[1]);
                             auto max  = std::max(a[2],b[2]);
                             Reduce_type red{ plus, min, max };
                             return red; 
                        },
                        [=](Index_type i) -> std::array<Int_type,3>{
                             Reduce_type val{ vec[i], vec[i], vec[i] };
                             return val; 

                        }
        );

        m_vsum += result[0];
        m_vmin = std::min(m_vmin, result[1]);
        m_vmax = std::max(m_vmax, result[2]);

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto init3_base_lam = [=](Index_type i) -> Int_type {
                              return vec[i];
                            };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          vsum += init3_base_lam(i);
          vmin = std::min(vmin, init3_base_lam(i));
          vmax = std::max(vmax, init3_base_lam(i));
        }

        m_vsum += vsum;
        m_vmin = std::min(m_vmin, vmin);
        m_vmax = std::max(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  REDUCE3_INT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

