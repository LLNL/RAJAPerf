//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf
{
namespace polybench
{


void POLYBENCH_MVT::runSeqVariant(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_MVT_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_MVT_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_MVT_BODY2;
          }
          POLYBENCH_MVT_BODY3;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_MVT_BODY4;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_MVT_BODY5;
          }
          POLYBENCH_MVT_BODY6;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto poly_mvt_base_lam2 = [=] (Index_type i, Index_type j,
                                     Real_type &dot) {
                                  POLYBENCH_MVT_BODY2;
                                 };
      auto poly_mvt_base_lam3 = [=] (Index_type i,
                                     Real_type &dot) {
                                  POLYBENCH_MVT_BODY3;
                                };
      auto poly_mvt_base_lam5 = [=] (Index_type i, Index_type j,
                                     Real_type &dot) {
                                  POLYBENCH_MVT_BODY5;
                                };
      auto poly_mvt_base_lam6 = [=] (Index_type i,
                                     Real_type &dot) {
                                  POLYBENCH_MVT_BODY6;
                                };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_MVT_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            poly_mvt_base_lam2(i, j, dot);
          }
          poly_mvt_base_lam3(i, dot);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_MVT_BODY4;
          for (Index_type j = 0; j < N; ++j ) {
            poly_mvt_base_lam5(i, j, dot);
          }
          poly_mvt_base_lam6(i, dot);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      POLYBENCH_MVT_VIEWS_RAJA;

      auto poly_mvt_lam1 = [=] (Real_type &dot) {
                                POLYBENCH_MVT_BODY1_RAJA;
                               };
      auto poly_mvt_lam2 = [=] (Index_type i, Index_type j, Real_type &dot) {
                                POLYBENCH_MVT_BODY2_RAJA;
                               };
      auto poly_mvt_lam3 = [=] (Index_type i, Real_type &dot) {
                                POLYBENCH_MVT_BODY3_RAJA;
                               };
      auto poly_mvt_lam4 = [=] (Real_type &dot) {
                                POLYBENCH_MVT_BODY4_RAJA;
                               };
      auto poly_mvt_lam5 = [=] (Index_type i, Index_type j, Real_type &dot) {
                                POLYBENCH_MVT_BODY5_RAJA;
                               };
      auto poly_mvt_lam6 = [=] (Index_type i, Real_type &dot) {
                                POLYBENCH_MVT_BODY6_RAJA;
                               };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::seq_exec,    // i
            RAJA::statement::Lambda<0, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,  // j
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
          RAJA::kernel_param_resource<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::tuple<Real_type>{0.0},
            res,

            poly_mvt_lam1,
            poly_mvt_lam2,
            poly_mvt_lam3

          );
          RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

          RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
          RAJA::kernel_param_resource<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::tuple<Real_type>{0.0},
            res,

            poly_mvt_lam4,
            poly_mvt_lam5,
            poly_mvt_lam6

          );
          RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

        }); // end sequential region (for single-source code)

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  POLYBENCH_MVT : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_MVT, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace polybench
} // end namespace rajaperf
