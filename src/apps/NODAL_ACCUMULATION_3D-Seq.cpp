//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void NODAL_ACCUMULATION_3D::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          NODAL_ACCUMULATION_3D_BODY_INDEX;
          NODAL_ACCUMULATION_3D_BODY(RAJAPERF_ATOMIC_ADD_SEQ);
        }
        RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto nodal_accumulation_3d_lam = [=](Index_type ii) {
                         NODAL_ACCUMULATION_3D_BODY_INDEX;
                         NODAL_ACCUMULATION_3D_BODY(RAJAPERF_ATOMIC_ADD_SEQ);
                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          nodal_accumulation_3d_lam(ii);
        }
        RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};
      RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                               res, RAJA::Unowned);

      auto nodal_accumulation_3d_lam = [=](Index_type i) {
                         NODAL_ACCUMULATION_3D_BODY(RAJAPERF_ATOMIC_ADD_RAJA_SEQ);
                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        RAJA::forall<RAJA::seq_exec>(res, zones, nodal_accumulation_3d_lam);
        RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(NODAL_ACCUMULATION_3D, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
