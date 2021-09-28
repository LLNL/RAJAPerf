//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#include <array>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void REDUCE3_INT::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        typedef std::array<Int_type,3> Reduce_type;
        Reduce_type result =
        std::transform_reduce( std::execution::par_unseq,
                                      std::begin(range), std::end(range),
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
        m_vmin = RAJA_MIN(m_vmin, result[1]);
        m_vmax = RAJA_MAX(m_vmax, result[2]);

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
          vmin = RAJA_MIN(vmin, init3_base_lam(i));
          vmax = RAJA_MAX(vmax, init3_base_lam(i));
        }

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Int_type> vsum(m_vsum_init);
        RAJA::ReduceMin<RAJA::seq_reduce, Int_type> vmin(m_vmin_init);
        RAJA::ReduceMax<RAJA::seq_reduce, Int_type> vmax(m_vmax_init);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          REDUCE3_INT_BODY_RAJA;
        });

        m_vsum += static_cast<Int_type>(vsum.get());
        m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
        m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  REDUCE3_INT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
