//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void NODAL_ACCUMULATION_3D::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#warning needs parallel for+atomic or reduce
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          NODAL_ACCUMULATION_3D_BODY_INDEX;
          NODAL_ACCUMULATION_3D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto nodal_accumulation_3d_lam = [=](Index_type ii) {
                         NODAL_ACCUMULATION_3D_BODY_INDEX;
                         NODAL_ACCUMULATION_3D_BODY;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#warning needs parallel for+atomic or reduce
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          nodal_accumulation_3d_lam(ii);
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      camp::resources::Resource working_res{camp::resources::Host()};
      RAJA::TypedListSegment<Index_type> zones(m_domain->real_zones,
                                               m_domain->n_real_zones,
                                               working_res);

      auto nodal_accumulation_3d_lam = [=](Index_type i) {
                         NODAL_ACCUMULATION_3D_RAJA_ATOMIC_BODY(RAJA::seq_atomic);
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(zones, nodal_accumulation_3d_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf