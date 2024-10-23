//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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


void MATVEC_3D_STENCIL::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  MATVEC_3D_STENCIL_DATA_SETUP;


  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          MATVEC_3D_STENCIL_BODY_INDEX;
          MATVEC_3D_STENCIL_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto matvec_3d_lam = [=](Index_type ii) {
            MATVEC_3D_STENCIL_BODY_INDEX;
            MATVEC_3D_STENCIL_BODY;
          };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          matvec_3d_lam(ii);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                               res, RAJA::Unowned);

      auto matvec_3d_lam = [=](Index_type i) {
                                         MATVEC_3D_STENCIL_BODY;
                                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          zones, matvec_3d_lam);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  MATVEC_3D_STENCIL : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
