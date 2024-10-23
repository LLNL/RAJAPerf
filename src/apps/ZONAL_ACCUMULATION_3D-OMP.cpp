//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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


void ZONAL_ACCUMULATION_3D::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  ZONAL_ACCUMULATION_3D_DATA_SETUP;


  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          ZONAL_ACCUMULATION_3D_BODY_INDEX;
          ZONAL_ACCUMULATION_3D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto zonal_accumulation_3d_lam = [=](Index_type ii) {
            ZONAL_ACCUMULATION_3D_BODY_INDEX;
            ZONAL_ACCUMULATION_3D_BODY;
          };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          zonal_accumulation_3d_lam(ii);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                               res, RAJA::Unowned);

      auto zonal_accumulation_3d_lam = [=](Index_type i) {
                                         ZONAL_ACCUMULATION_3D_BODY;
                                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          zones, zonal_accumulation_3d_lam);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  ZONAL_ACCUMULATION_3D : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
