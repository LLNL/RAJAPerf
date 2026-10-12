//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <cstring>


namespace rajaperf
{
namespace polybench
{


void POLYBENCH_GEMVER::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMVER_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMVER_BODY1;
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY2;
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMVER_BODY3;
          }
          POLYBENCH_GEMVER_BODY4;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY5;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY6;
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMVER_BODY7;
          }
          POLYBENCH_GEMVER_BODY8;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto poly_gemver_base_lam1 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_GEMVER_BODY1;
                                   };
      auto poly_gemver_base_lam3 = [=](Index_type i, Index_type j,
                                       Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY3;
                                   };
      auto poly_gemver_base_lam4 = [=](Index_type i, Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY4;
                                   };
      auto poly_gemver_base_lam5 = [=](Index_type i) {
                                     POLYBENCH_GEMVER_BODY5;
                                   };
      auto poly_gemver_base_lam7 = [=](Index_type i, Index_type j,
                                       Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY7;
                                    };
      auto poly_gemver_base_lam8 = [=](Index_type i, Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY8;
                                   };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            poly_gemver_base_lam1(i, j);
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY2;
          for (Index_type j = 0; j < n; j++) {
            poly_gemver_base_lam3(i, j, dot);
          }
          poly_gemver_base_lam4(i, dot);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          poly_gemver_base_lam5(i);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY6;
          for (Index_type j = 0; j < n; j++) {
            poly_gemver_base_lam7(i, j, dot);
          }
          poly_gemver_base_lam8(i, dot);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      POLYBENCH_GEMVER_VIEWS_RAJA;

      auto poly_gemver_lam1 = [=] (Index_type i, Index_type j) {
                                   POLYBENCH_GEMVER_BODY1_RAJA;
                                  };
      auto poly_gemver_lam2 = [=] (Index_type /* i */, Real_type &dot) {
                                   POLYBENCH_GEMVER_BODY2_RAJA;
                                  };
      auto poly_gemver_lam3 = [=] (Index_type i, Index_type j, Real_type &dot) {
                                   POLYBENCH_GEMVER_BODY3_RAJA;
                                  };
      auto poly_gemver_lam4 = [=] (Index_type i, Real_type &dot) {
                                   POLYBENCH_GEMVER_BODY4_RAJA;
                                  };
      auto poly_gemver_lam5 = [=] (Index_type i) {
                                   POLYBENCH_GEMVER_BODY5_RAJA;
                                  };
      auto poly_gemver_lam6 = [=] (Index_type i, Real_type &dot) {
                                   POLYBENCH_GEMVER_BODY6_RAJA;
                                  };
      auto poly_gemver_lam7 = [=] (Index_type i, Index_type j, Real_type &dot) {
                                   POLYBENCH_GEMVER_BODY7_RAJA;
                                  };
      auto poly_gemver_lam8 = [=] (Index_type i, Real_type &dot) {
                                   POLYBENCH_GEMVER_BODY8_RAJA;
                                  };

      using EXEC_POL1 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<0, RAJA::Segs<0,1>>
            >
          >
        >;

      using EXEC_POL24 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >;

      using EXEC_POL3 = RAJA::seq_exec;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
        RAJA::kernel_resource<EXEC_POL1>(
          RAJA::make_tuple(RAJA::RangeSegment{0, n},
                           RAJA::RangeSegment{0, n}),
          res,
          poly_gemver_lam1
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
        RAJA::kernel_param_resource<EXEC_POL24>(
          RAJA::make_tuple(RAJA::RangeSegment{0, n},
                           RAJA::RangeSegment{0, n}),
          RAJA::tuple<Real_type>{0.0},
          res,

          poly_gemver_lam2,
          poly_gemver_lam3,
          poly_gemver_lam4
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
        RAJA::forall<EXEC_POL3>( res,
          RAJA::RangeSegment{0, n},
          poly_gemver_lam5
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
        RAJA::kernel_param_resource<EXEC_POL24>(
          RAJA::make_tuple(RAJA::RangeSegment{0, n},
                           RAJA::RangeSegment{0, n}),
          RAJA::tuple<Real_type>{0.0},
          res,

          poly_gemver_lam6,
          poly_gemver_lam7,
          poly_gemver_lam8

        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_GEMVER : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GEMVER, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace basic
} // end namespace rajaperf
