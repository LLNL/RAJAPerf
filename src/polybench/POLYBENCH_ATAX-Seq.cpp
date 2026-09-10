//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf
{
namespace polybench
{

void POLYBENCH_ATAX::runSeqVariant(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_ATAX_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_1");
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_ATAX_BODY2;
          }
          POLYBENCH_ATAX_BODY3;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_2");
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY4;
          for (Index_type i = 0; i < N; ++i ) {
            POLYBENCH_ATAX_BODY5;
          }
          POLYBENCH_ATAX_BODY6;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_2");

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto poly_atax_base_lam2 = [=] (Index_type i, Index_type j,
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY2;
                                 };
      auto poly_atax_base_lam3 = [=] (Index_type i,
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY3;
                                  };
      auto poly_atax_base_lam5 = [=] (Index_type i, Index_type j ,
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY5;
                                  };
      auto poly_atax_base_lam6 = [=] (Index_type j,
                                      Real_type &dot) {
                                   POLYBENCH_ATAX_BODY6;
                                  };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_1");
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            poly_atax_base_lam2(i, j, dot);
          }
          poly_atax_base_lam3(i, dot);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_2");
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY4;
          for (Index_type i = 0; i < N; ++i ) {
            poly_atax_base_lam5(i, j, dot);
          }
          poly_atax_base_lam6(j, dot);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_2");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      POLYBENCH_ATAX_VIEWS_RAJA;

      auto poly_atax_lam1 = [=] (Index_type i, Real_type &dot) {
                              POLYBENCH_ATAX_BODY1_RAJA;
                             };
      auto poly_atax_lam2 = [=] (Index_type i, Index_type j, Real_type &dot) {
                              POLYBENCH_ATAX_BODY2_RAJA;
                             };
      auto poly_atax_lam3 = [=] (Index_type i, Real_type &dot) {
                              POLYBENCH_ATAX_BODY3_RAJA;
                             };
      auto poly_atax_lam4 = [=] (Index_type j, Real_type &dot) {
                              POLYBENCH_ATAX_BODY4_RAJA;
                             };
      auto poly_atax_lam5 = [=] (Index_type i, Index_type j , Real_type &dot) {
                              POLYBENCH_ATAX_BODY5_RAJA;
                             };
      auto poly_atax_lam6 = [=] (Index_type j, Real_type &dot) {
                              POLYBENCH_ATAX_BODY6_RAJA;
                             };

      using EXEC_POL1 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::seq_exec,
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >;

      using EXEC_POL2 =
        RAJA::KernelPolicy<
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<0, RAJA::Segs<1>, RAJA::Params<0>>,
            RAJA::statement::For<0, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<1>, RAJA::Params<0>>
          >
        >;


      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_1");
        RAJA::kernel_param_resource<EXEC_POL1>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},
          res,

          poly_atax_lam1,
          poly_atax_lam2,
          poly_atax_lam3

        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_2");
        RAJA::kernel_param_resource<EXEC_POL2>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},
          res,

          poly_atax_lam4,
          poly_atax_lam5,
          poly_atax_lam6

        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_2");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  POLYBENCH_ATAX : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_ATAX, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace polybench
} // end namespace rajaperf
