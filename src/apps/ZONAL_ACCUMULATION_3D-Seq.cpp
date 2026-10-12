//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ZONAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void ZONAL_ACCUMULATION_3D::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  ZONAL_ACCUMULATION_3D_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ZONAL_ACCUMULATION_3D_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          ZONAL_ACCUMULATION_3D_BODY_INDEX;
          ZONAL_ACCUMULATION_3D_BODY;
        }
        RP_CALI_SUBKERNEL_END("ZONAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto zonal_accumulation_3d_lam = [=](Index_type ii) {
                         ZONAL_ACCUMULATION_3D_BODY_INDEX;
                         ZONAL_ACCUMULATION_3D_BODY;
                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ZONAL_ACCUMULATION_3D_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          zonal_accumulation_3d_lam(ii);
        }
        RP_CALI_SUBKERNEL_END("ZONAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                               res, RAJA::Unowned);

      auto zonal_accumulation_3d_lam = [=](Index_type i) {
                         ZONAL_ACCUMULATION_3D_BODY;
                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ZONAL_ACCUMULATION_3D_1");
        RAJA::forall<RAJA::seq_exec>(res, zones, zonal_accumulation_3d_lam);
        RP_CALI_SUBKERNEL_END("ZONAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  ZONAL_ACCUMULATION_3D : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(ZONAL_ACCUMULATION_3D, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
