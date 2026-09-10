//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <limits>
#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t tune_idx >
void REDUCE3_INT::runOpenMPVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("REDUCE3_INT_1");
        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        #pragma omp parallel for reduction(+:vsum), \
                                 reduction(min:vmin), \
                                 reduction(max:vmax)
        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE3_INT_BODY;
        }

        m_vsum = vsum;
        m_vmin = vmin;
        m_vmax = vmax;
        RP_CALI_SUBKERNEL_END("REDUCE3_INT_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto reduce3int_base_lam = [=](Index_type i) -> Int_type {
                                   return vec[i];
                                 };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("REDUCE3_INT_1");
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

        m_vsum = vsum;
        m_vmin = vmin;
        m_vmax = vmax;
        RP_CALI_SUBKERNEL_END("REDUCE3_INT_1");

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

          RP_CALI_SUBKERNEL_BEGIN("REDUCE3_INT_1");
          RAJA::ReduceSum<RAJA::omp_reduce, Int_type> vsum(m_vsum_init);
          RAJA::ReduceMin<RAJA::omp_reduce, Int_type> vmin(m_vmin_init);
          RAJA::ReduceMax<RAJA::omp_reduce, Int_type> vmax(m_vmax_init);
  
          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            REDUCE3_INT_BODY_RAJA;
          });

          m_vsum = static_cast<Int_type>(vsum.get());
          m_vmin = static_cast<Int_type>(vmin.get());
          m_vmax = static_cast<Int_type>(vmax.get());
          RP_CALI_SUBKERNEL_END("REDUCE3_INT_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE3_INT_1");
          Int_type tvsum = m_vsum_init;
          Int_type tvmin = m_vmin_init;
          Int_type tvmax = m_vmax_init;

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tvsum),
            RAJA::expt::Reduce<RAJA::operators::minimum>(&tvmin),
            RAJA::expt::Reduce<RAJA::operators::maximum>(&tvmax),
            [=](Index_type i,
              RAJA::expt::ValOp<Int_type, RAJA::operators::plus>& vsum,
              RAJA::expt::ValOp<Int_type, RAJA::operators::minimum>& vmin,
              RAJA::expt::ValOp<Int_type, RAJA::operators::maximum>& vmax) { 
              REDUCE3_INT_BODY_RAJA;
            }
          );

          m_vsum = static_cast<Int_type>(tvsum);
          m_vmin = static_cast<Int_type>(tvmin);
          m_vmax = static_cast<Int_type>(tvmax);
          RP_CALI_SUBKERNEL_END("REDUCE3_INT_1");

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

}

void REDUCE3_INT::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    addVariantTuning<&REDUCE3_INT::runOpenMPVariant<0>>(
        vid, "default");

    if (vid == RAJA_OpenMP) {

      addVariantTuning<&REDUCE3_INT::runOpenMPVariant<1>>(
          vid, "new");

    }

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif
