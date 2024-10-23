//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void LTIMES_NOVIEW::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  LTIMES_NOVIEW_DATA_SETUP;

  auto ltimesnoview_lam = [=](Index_type d, Index_type z,
                              Index_type g, Index_type m) {
                                LTIMES_NOVIEW_BODY;
                          };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type z = 0; z < num_z; ++z ) {
          for (Index_type g = 0; g < num_g; ++g ) {
            for (Index_type m = 0; m < num_m; ++m ) {
              for (Index_type d = 0; d < num_d; ++d ) {
                LTIMES_NOVIEW_BODY;
              }
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type z = 0; z < num_z; ++z ) {
          for (Index_type g = 0; g < num_g; ++g ) {
            for (Index_type m = 0; m < num_m; ++m ) {
              for (Index_type d = 0; d < num_d; ++d ) {
                ltimesnoview_lam(d, z, g, m);
              }
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<1, RAJA::omp_parallel_for_exec, // z
            RAJA::statement::For<2, RAJA::seq_exec,            // g
              RAJA::statement::For<3, RAJA::seq_exec,          // m
                RAJA::statement::For<0, RAJA::seq_exec,        // d
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, num_d),
                                                          RAJA::RangeSegment(0, num_z),
                                                          RAJA::RangeSegment(0, num_g),
                                                          RAJA::RangeSegment(0, num_m)),
                                         res,
                                         ltimesnoview_lam
                                       );

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n LTIMES_NOVIEW : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
