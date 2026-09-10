//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{


void POLYBENCH_FDTD_2D::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_1");
        for (Index_type j = 0; j < ny; j++) {
          POLYBENCH_FDTD_2D_BODY1;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_1");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_2");
        for (Index_type i = 1; i < nx; i++) {
          for (Index_type j = 0; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY2;
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_2");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_3");
        for (Index_type i = 0; i < nx; i++) {
          for (Index_type j = 1; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY3;
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_3");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_4");
        for (Index_type i = 0; i < nx - 1; i++) {
          for (Index_type j = 0; j < ny - 1; j++) {
            POLYBENCH_FDTD_2D_BODY4;
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_4");

        t = (t+1) % m_tsteps;
      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      //
      // Note: first lambda must use capture by reference so that the
      //       scalar variable 't' used in it is updated for each
      //       t-loop iteration.
      //
      auto poly_fdtd2d_base_lam1 = [&](Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY1;
                                   };
      auto poly_fdtd2d_base_lam2 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY2;
                                   };
      auto poly_fdtd2d_base_lam3 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY3;
                                   };
      auto poly_fdtd2d_base_lam4 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_FDTD_2D_BODY4;
                                   };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_1");
        for (Index_type j = 0; j < ny; j++) {
          poly_fdtd2d_base_lam1(j);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_1");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_2");
        for (Index_type i = 1; i < nx; i++) {
          for (Index_type j = 0; j < ny; j++) {
            poly_fdtd2d_base_lam2(i, j);
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_2");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_3");
        for (Index_type i = 0; i < nx; i++) {
          for (Index_type j = 1; j < ny; j++) {
            poly_fdtd2d_base_lam3(i, j);
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_3");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_4");
        for (Index_type i = 0; i < nx - 1; i++) {
          for (Index_type j = 0; j < ny - 1; j++) {
            poly_fdtd2d_base_lam4(i, j);
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_4");

        t = (t+1) % m_tsteps;
      }  // run_reps
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      POLYBENCH_FDTD_2D_VIEWS_RAJA;

      //
      // Note: first lambda must use capture by reference so that the
      //       scalar variable 't' used in it is updated for each
      //       t-loop iteration.
      //
      auto poly_fdtd2d_lam1 = [&](Index_type j) {
                                POLYBENCH_FDTD_2D_BODY1_RAJA;
                              };
      auto poly_fdtd2d_lam2 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY2_RAJA;
                              };
      auto poly_fdtd2d_lam3 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY3_RAJA;
                              };
      auto poly_fdtd2d_lam4 = [=](Index_type i, Index_type j) {
                                POLYBENCH_FDTD_2D_BODY4_RAJA;
                              };

      using EXEC_POL1 = RAJA::seq_exec;

      using EXEC_POL234 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::seq_exec,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_1");
        RAJA::forall<EXEC_POL1>(res,
           RAJA::RangeSegment(0, ny),
          poly_fdtd2d_lam1
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_2");
        RAJA::kernel_resource<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                           RAJA::RangeSegment{0, ny}),
          res,
          poly_fdtd2d_lam2
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_2");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_3");
        RAJA::kernel_resource<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                           RAJA::RangeSegment{1, ny}),
          res,
          poly_fdtd2d_lam3
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_3");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_4");
        RAJA::kernel_resource<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                           RAJA::RangeSegment{0, ny-1}),
          res,
          poly_fdtd2d_lam4
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_4");

        t = (t+1) % m_tsteps;
      } // run_reps
      stopTimer();

      break;
    }

#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\nPOLYBENCH_FDTD_2D  Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_FDTD_2D, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace polybench
} // end namespace rajaperf
