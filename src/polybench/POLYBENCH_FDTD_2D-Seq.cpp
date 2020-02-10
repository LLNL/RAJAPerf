//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

namespace rajaperf 
{
namespace polybench
{


void POLYBENCH_FDTD_2D::runSeqVariant(VariantID vid)
{

  runKernel(vid);

#if 0
  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  // IMPORTANT: This first lambda definition must use capture by reference to
  //            get the correct result (the scalar vaiable 't' used in
  //            the lambda expression gets modified by the kernel).
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

  POLYBENCH_FDTD_2D_VIEWS_RAJA;

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

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) { 

          for (Index_type j = 0; j < ny; j++) {
            POLYBENCH_FDTD_2D_BODY1;
          }
          for (Index_type i = 1; i < nx; i++) {
            for (Index_type j = 0; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY2;
            }
          }
          for (Index_type i = 0; i < nx; i++) {
            for (Index_type j = 1; j < ny; j++) {
              POLYBENCH_FDTD_2D_BODY3;
            }
          }
          for (Index_type i = 0; i < nx - 1; i++) {
            for (Index_type j = 0; j < ny - 1; j++) {
              POLYBENCH_FDTD_2D_BODY4;
            }
          }

        }  // tstep loop

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) {

          for (Index_type j = 0; j < ny; j++) {
            poly_fdtd2d_base_lam1(j);
          }
          for (Index_type i = 1; i < nx; i++) {
            for (Index_type j = 0; j < ny; j++) {
              poly_fdtd2d_base_lam2(i, j);
            }
          }
          for (Index_type i = 0; i < nx; i++) {
            for (Index_type j = 1; j < ny; j++) {
              poly_fdtd2d_base_lam3(i, j);
            }
          }
          for (Index_type i = 0; i < nx - 1; i++) {
            for (Index_type j = 0; j < ny - 1; j++) {
              poly_fdtd2d_base_lam4(i, j);
            }
          }

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      using EXEC_POL1 = RAJA::loop_exec;

      using EXEC_POL234 =  
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (t = 0; t < tsteps; ++t) { 

          RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny), 
            poly_fdtd2d_lam1
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                             RAJA::RangeSegment{0, ny}),
            poly_fdtd2d_lam2
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                             RAJA::RangeSegment{1, ny}),
            poly_fdtd2d_lam3
          );

          RAJA::kernel<EXEC_POL234>(
            RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                             RAJA::RangeSegment{0, ny-1}),
            poly_fdtd2d_lam4
          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\nPOLYBENCH_FDTD_2D  Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
