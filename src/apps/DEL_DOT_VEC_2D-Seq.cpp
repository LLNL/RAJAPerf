//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include "camp/resource.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void DEL_DOT_VEC_2D::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  DEL_DOT_VEC_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DEL_DOT_VEC_2D_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          DEL_DOT_VEC_2D_BODY_INDEX;
          DEL_DOT_VEC_2D_BODY;
        }
        RP_CALI_SUBKERNEL_END("DEL_DOT_VEC_2D_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto deldotvec2d_base_lam = [=](Index_type ii) {
                                    DEL_DOT_VEC_2D_BODY_INDEX;
                                    DEL_DOT_VEC_2D_BODY;
                                  };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DEL_DOT_VEC_2D_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          deldotvec2d_base_lam(ii);
        }
        RP_CALI_SUBKERNEL_END("DEL_DOT_VEC_2D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                               res, RAJA::Unowned);

      auto deldotvec2d_lam = [=](Index_type i) {
                               DEL_DOT_VEC_2D_BODY;
                             };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DEL_DOT_VEC_2D_1");
        RAJA::forall<RAJA::seq_exec>(res, zones, deldotvec2d_lam);
        RP_CALI_SUBKERNEL_END("DEL_DOT_VEC_2D_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  DEL_DOT_VEC_2D : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DEL_DOT_VEC_2D, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
