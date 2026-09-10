//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXHEX.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>



namespace rajaperf
{
namespace apps
{



void INTSC_HEXHEX::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0 ;
  const Index_type iend = m_nthreads ;

  const Index_type n_szpairs   = m_n_subz_intsc ;

  INTSC_HEXHEX_DATA_SETUP ;

#if defined(RUN_RAJA_SEQ)
  auto intsc_hexhex_lam = [=](Index_type i) {
      INTSC_HEXHEX_SEQ ( i, iend ) ;
  } ;
  auto fixup_vv_lam     = [=](Index_type i) {
      FIXUP_VV_BODY ;
  } ;
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_1");
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          INTSC_HEXHEX_SEQ ( i, iend ) ;
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_1");
        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_2");
        for (Index_type i = ibegin ; i < n_szpairs ; ++i ) {
          FIXUP_VV_BODY ;
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_2");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_1");
        for (Index_type i = ibegin ; i < iend; ++i ) {
          intsc_hexhex_lam( i );
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_1");
        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_2");
        for (Index_type i = ibegin ; i < n_szpairs ; ++i ) {
          fixup_vv_lam( i );
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_2");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend), intsc_hexhex_lam);
        RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_1");
        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_2");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, n_szpairs), fixup_vv_lam);
        RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_2");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  INTSC_HEXHEX : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INTSC_HEXHEX, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
