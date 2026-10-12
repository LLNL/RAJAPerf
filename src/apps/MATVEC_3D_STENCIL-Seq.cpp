//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MATVEC_3D_STENCIL.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void MATVEC_3D_STENCIL::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  MATVEC_3D_STENCIL_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MATVEC_3D_STENCIL_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          MATVEC_3D_STENCIL_BODY_INDEX;
          MATVEC_3D_STENCIL_BODY;
        }
        RP_CALI_SUBKERNEL_END("MATVEC_3D_STENCIL_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto matvec_3d_lam = [=](Index_type ii) {
                         MATVEC_3D_STENCIL_BODY_INDEX;
                         MATVEC_3D_STENCIL_BODY;
                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MATVEC_3D_STENCIL_1");
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          matvec_3d_lam(ii);
        }
        RP_CALI_SUBKERNEL_END("MATVEC_3D_STENCIL_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};
      RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                               res, RAJA::Unowned);

      auto matvec_3d_lam = [=](Index_type i) {
                         MATVEC_3D_STENCIL_BODY;
                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MATVEC_3D_STENCIL_1");
        RAJA::forall<RAJA::seq_exec>(res, zones, matvec_3d_lam);
        RP_CALI_SUBKERNEL_END("MATVEC_3D_STENCIL_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  MATVEC_3D_STENCIL : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MATVEC_3D_STENCIL, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
