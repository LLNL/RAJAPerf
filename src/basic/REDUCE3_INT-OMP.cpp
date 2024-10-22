//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#include <limits>
#include <iostream>

namespace rajaperf
{
namespace basic
{


void REDUCE3_INT::runOpenMPVariant(VariantID vid, size_t tune_idx)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        #pragma omp parallel for reduction(+:vsum), \
                                 reduction(min:vmin), \
                                 reduction(max:vmax)
        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE3_INT_BODY;
        }

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto reduce3int_base_lam = [=](Index_type i) -> Int_type {
                                   return vec[i];
                                 };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        #pragma omp parallel for reduction(+:vsum), \
                                 reduction(min:vmin), \
                                 reduction(max:vmax)
        for (Index_type i = ibegin; i < iend; ++i ) {
          vsum += reduce3int_base_lam(i);
          vmin = RAJA_MIN(vmin, reduce3int_base_lam(i));
          vmax = RAJA_MAX(vmax, reduce3int_base_lam(i));
        }

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      if (tune_idx == 0) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          RAJA::ReduceSum<RAJA::omp_reduce, Int_type> vsum(m_vsum_init);
          RAJA::ReduceMin<RAJA::omp_reduce, Int_type> vmin(m_vmin_init);
          RAJA::ReduceMax<RAJA::omp_reduce, Int_type> vmax(m_vmax_init);
  
          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            REDUCE3_INT_BODY_RAJA;
          });

          m_vsum += static_cast<Int_type>(vsum.get());
          m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
          m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

        }
        stopTimer();

      } else if (tune_idx == 1) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          Int_type tvsum = m_vsum_init;
          Int_type tvmin = m_vmin_init;
          Int_type tvmax = m_vmax_init;

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tvsum),
            RAJA::expt::Reduce<RAJA::operators::minimum>(&tvmin),
            RAJA::expt::Reduce<RAJA::operators::maximum>(&tvmax),
            [=](Index_type i, Int_type& vsum, Int_type& vmin, Int_type& vmax) {
              REDUCE3_INT_BODY;
            }
          );

          m_vsum += static_cast<Int_type>(tvsum);
          m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(tvmin));
          m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(tvmax));

        }
        stopTimer();

      } else {
        getCout() << "\n  REDUCE3_INT : Unknown OpenMP tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  REDUCE3_INT : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
  RAJA_UNUSED_VAR(tune_idx);
#endif
}

void REDUCE3_INT::setOpenMPTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_OpenMP) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace basic
} // end namespace rajaperf
