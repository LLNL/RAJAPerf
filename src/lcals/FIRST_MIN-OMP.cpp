//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{


void FIRST_MIN::runOpenMPVariant(VariantID vid, size_t tune_idx)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_MIN_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp declare reduction(minloc : MyMinLoc : \
                                      omp_out = MinLoc_compare(omp_out, omp_in)) \
                                      initializer (omp_priv = omp_orig)

        FIRST_MIN_MINLOC_INIT;

        #pragma omp parallel for reduction(minloc:mymin)
        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_MIN_BODY;
        }

        m_minloc = mymin.loc;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto firstmin_base_lam = [=](Index_type i) -> Real_type {
                                 return x[i];
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

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

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      if (tune_idx == 0) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
          RAJA::ReduceMinLoc<RAJA::omp_reduce, Real_type, Index_type> loc(
                                                          m_xmin_init, m_initloc);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            FIRST_MIN_BODY_RAJA;
          });

          m_minloc = loc.getLoc();

        }
        stopTimer();

      } else if (tune_idx == 1) {

        using VL_TYPE = RAJA::expt::ValLoc<Real_type>;

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          VL_TYPE tloc(m_xmin_init, m_initloc);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::minimum>(&tloc),
            [=](Index_type i, VL_TYPE& loc) {
              loc.min(x[i], i);
            }
          );

          m_minloc = static_cast<Index_type>(tloc.getLoc());

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

#else
  RAJA_UNUSED_VAR(vid);
  RAJA_UNUSED_VAR(tune_idx);
#endif
}

void FIRST_MIN::setOpenMPTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_OpenMP) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace lcals
} // end namespace rajaperf
