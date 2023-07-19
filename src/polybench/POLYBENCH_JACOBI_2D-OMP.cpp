//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf
{
namespace polybench
{


void POLYBENCH_JACOBI_2D::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              POLYBENCH_JACOBI_2D_BODY1;
            }
          }

          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              POLYBENCH_JACOBI_2D_BODY2;
            }
          }

        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto poly_jacobi2d_base_lam1 = [=](Index_type i, Index_type j) {
                                       POLYBENCH_JACOBI_2D_BODY1;
                                     };
      auto poly_jacobi2d_base_lam2 = [=](Index_type i, Index_type j) {
                                       POLYBENCH_JACOBI_2D_BODY2;
                                     };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              poly_jacobi2d_base_lam1(i, j);
            }
          }

          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            for (Index_type j = 1; j < N-1; ++j ) {
              poly_jacobi2d_base_lam2(i, j);
            }
          }

        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_JACOBI_2D_VIEWS_RAJA;

      auto poly_jacobi2d_lam1 = [=](Index_type i, Index_type j) {
                                  POLYBENCH_JACOBI_2D_BODY1_RAJA;
                                };
      auto poly_jacobi2d_lam2 = [=](Index_type i, Index_type j) {
                                  POLYBENCH_JACOBI_2D_BODY2_RAJA;
                                };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<0>
            >
          >,
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

	    poly_jacobi2d_lam1,
	    poly_jacobi2d_lam2
          );

        }

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace polybench
} // end namespace rajaperf
