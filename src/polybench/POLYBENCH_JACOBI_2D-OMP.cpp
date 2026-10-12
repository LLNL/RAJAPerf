//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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


void POLYBENCH_JACOBI_2D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
        #pragma omp parallel for
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            POLYBENCH_JACOBI_2D_BODY1;
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
        #pragma omp parallel for
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            POLYBENCH_JACOBI_2D_BODY2;
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

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
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
        #pragma omp parallel for
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            poly_jacobi2d_base_lam1(i, j);
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
        #pragma omp parallel for
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            poly_jacobi2d_base_lam2(i, j);
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

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
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                           RAJA::RangeSegment{1, N-1}),
          res,
          poly_jacobi2d_lam1
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                           RAJA::RangeSegment{1, N-1}),
          res,
          poly_jacobi2d_lam2
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

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

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_2D, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace polybench
} // end namespace rajaperf
