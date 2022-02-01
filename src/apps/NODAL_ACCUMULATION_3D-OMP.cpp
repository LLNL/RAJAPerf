//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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


void NODAL_ACCUMULATION_3D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;


  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          NODAL_ACCUMULATION_3D_BODY_INDEX;

          Real_type val = 0.125 * vol[i];

          #pragma omp atomic
          x0[i] += val;
          #pragma omp atomic
          x1[i] += val;
          #pragma omp atomic
          x2[i] += val;
          #pragma omp atomic
          x3[i] += val;
          #pragma omp atomic
          x4[i] += val;
          #pragma omp atomic
          x5[i] += val;
          #pragma omp atomic
          x6[i] += val;
          #pragma omp atomic
          x7[i] += val;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto nodal_accumulation_3d_lam = [=](Index_type ii) {
            NODAL_ACCUMULATION_3D_BODY_INDEX;

            Real_type val = 0.125 * vol[i];

            #pragma omp atomic
            x0[i] += val;
            #pragma omp atomic
            x1[i] += val;
            #pragma omp atomic
            x2[i] += val;
            #pragma omp atomic
            x3[i] += val;
            #pragma omp atomic
            x4[i] += val;
            #pragma omp atomic
            x5[i] += val;
            #pragma omp atomic
            x6[i] += val;
            #pragma omp atomic
            x7[i] += val;
          };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          nodal_accumulation_3d_lam(ii);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      camp::resources::Resource working_res{camp::resources::Host()};
      RAJA::TypedListSegment<Index_type> zones(m_domain->real_zones,
                                               m_domain->n_real_zones,
                                               working_res);

      auto nodal_accumulation_3d_lam = [=](Index_type i) {
                                         NODAL_ACCUMULATION_3D_RAJA_ATOMIC_BODY(RAJA::omp_atomic);
                                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          zones, nodal_accumulation_3d_lam);

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  NODAL_ACCUMULATION_3D : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
