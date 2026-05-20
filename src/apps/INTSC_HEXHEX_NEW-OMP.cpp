//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXHEX_NEW.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

void INTSC_HEXHEX_NEW::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  //  We compute each standard intersection within a single thread
  //  to avoid collisions in vv_out hence only distribute different
  //  standard intersections among threads, iend is getActualProblemSize.
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0 ;

  const Index_type n_subz_intsc= m_n_subz_intsc ;
  const Index_type n_szpairs   = m_n_subz_intsc;

  //  Thread loop is over grouped intersections between subzone pairs.
  const Index_type iend = RAJA_DIVIDE_CEILING_INT(n_subz_intsc, hexhex_new_fixup_groupsize);

  INTSC_HEXHEX_NEW_DATA_SETUP ;

  auto intsc_hexhex_lam = [=] ( Index_type i ) {
      INTSC_HEXHEX_NEW_OMP ( i, iend ) ;
  } ;
  auto fixup_vv_lam     = [=] ( Index_type i ) {
      INTSC_HEXHEX_NEW_FIXUP_VV_BODY ;
  } ;

  // Insert a warmup call to remove time of initialization of OpenMP
  // that affects the first call to the function.
  Bool_type const do_warmup = true ;
  if ( do_warmup ) {
#pragma omp parallel for
    for (Index_type i = ibegin ; i < iend ; ++i ) {
      INTSC_HEXHEX_NEW_OMP( i, iend ) ;
    }
#pragma omp parallel for
    for (Index_type i = ibegin ; i < n_szpairs ; ++i ) {
      INTSC_HEXHEX_NEW_FIXUP_VV_BODY ;
    }
  }

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          INTSC_HEXHEX_NEW_OMP( i, iend ) ;
        }
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < n_szpairs ; ++i ) {
          INTSC_HEXHEX_NEW_FIXUP_VV_BODY ;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          intsc_hexhex_lam(i);
        }
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < n_szpairs ; ++i ) {
          fixup_vv_lam( i ) ;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), intsc_hexhex_lam);
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, n_szpairs), fixup_vv_lam);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  INTSC_HEXHEX_NEW : Unknown OpenMP variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INTSC_HEXHEX_NEW, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace apps
} // end namespace rajaperf
