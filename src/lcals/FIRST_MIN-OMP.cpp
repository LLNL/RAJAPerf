//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <iostream>

namespace rajaperf
{
namespace lcals
{

template < size_t tune_idx >
void FIRST_MIN::runOpenMPVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_MIN_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIRST_MIN_1");
        #pragma omp declare reduction(minloc : MyMinLoc : \
                                      omp_out = MinLoc_compare(omp_out, omp_in)) \
                                      initializer (omp_priv = omp_orig)

        FIRST_MIN_MINLOC_INIT;

        #pragma omp parallel for reduction(minloc:mymin)
        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_MIN_BODY;
        }

        m_minloc = mymin.loc;
        RP_CALI_SUBKERNEL_END("FIRST_MIN_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto firstmin_base_lam = [=](Index_type i) -> Real_type {
                                 return x[i];
                               };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIRST_MIN_1");
        #pragma omp declare reduction(minloc : MyMinLoc : \
                                      omp_out = MinLoc_compare(omp_out, omp_in)) \
                                      initializer (omp_priv = omp_orig)

        FIRST_MIN_MINLOC_INIT;

        #pragma omp parallel for reduction(minloc:mymin)
        for (Index_type i = ibegin; i < iend; ++i ) {
          if ( firstmin_base_lam(i) < mymin.val ) {
            mymin.val = x[i];
            mymin.loc = i;
          }
        }

        m_minloc = mymin.loc;
        RP_CALI_SUBKERNEL_END("FIRST_MIN_1");

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

          RP_CALI_SUBKERNEL_BEGIN("FIRST_MIN_1");
          RAJA::ReduceMinLoc<RAJA::omp_reduce,
                             Real_type, Index_type> minloc(m_xmin_init,
                                                           m_initloc);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            FIRST_MIN_BODY_RAJA;
          });

          m_minloc = minloc.getLoc();
          RP_CALI_SUBKERNEL_END("FIRST_MIN_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("FIRST_MIN_1");
          RAJA::expt::ValLoc<Real_type, Index_type> tminloc(m_xmin_init,
                                                            m_initloc);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::minimum>(&tminloc),
            [=](Index_type i,
              RAJA::expt::ValLocOp<Real_type, Index_type,
                                   RAJA::operators::minimum>& minloc) {
              FIRST_MIN_BODY_RAJA;
            }
          );

          m_minloc = static_cast<Index_type>(tminloc.getLoc());
          RP_CALI_SUBKERNEL_END("FIRST_MIN_1");

        }
        stopTimer();

      } else {
        getCout() << "\n  FIRST_MIN : Unknown OpenMP tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  FIRST_MIN : Unknown variant id = " << vid << std::endl;
    }

  }

}

void FIRST_MIN::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    addVariantTuning<&FIRST_MIN::runOpenMPVariant<0>>(
        vid, "default");

    if (vid == RAJA_OpenMP) {

      addVariantTuning<&FIRST_MIN::runOpenMPVariant<1>>(
          vid, "new");

    }

  }

}

} // end namespace lcals
} // end namespace rajaperf

#endif
