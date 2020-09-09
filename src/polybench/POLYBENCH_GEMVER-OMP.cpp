//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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

  auto poly_gemver_base_lam1 = [=](Index_type i, Index_type j) {
                                 POLYBENCH_GEMVER_BODY1;
                               };
  auto poly_gemver_base_lam3 = [=](Index_type i, Index_type j, Real_type &dot) {
                                 POLYBENCH_GEMVER_BODY3;
                               };
  auto poly_gemver_base_lam4 = [=](Index_type i, Real_type &dot) {
                                 POLYBENCH_GEMVER_BODY4;
                               };
  auto poly_gemver_base_lam5 = [=](Index_type i) {
                                 POLYBENCH_GEMVER_BODY5;
                               };
  auto poly_gemver_base_lam7 = [=](Index_type i, Index_type j, Real_type &dot) {
                                 POLYBENCH_GEMVER_BODY7;
                                };
  auto poly_gemver_base_lam8 = [=](Index_type i, Real_type &dot) {
                                 POLYBENCH_GEMVER_BODY8;
                               };

  POLYBENCH_GEMVER_VIEWS_RAJA;

#if defined(RUN_RAJA_SEQ_ARGS) || defined(RUN_RAJA_SEQ_ARGS_DEV)
  auto poly_gemver_lam1 = [=] (Index_type i, Index_type j) {
                               POLYBENCH_GEMVER_BODY1_RAJA;
                              };
  auto poly_gemver_lam2 = [=] (Real_type &dot) {
                               POLYBENCH_GEMVER_BODY2_RAJA;
                              };
  auto poly_gemver_lam3 = [=] (Index_type i, Index_type j, Real_type &dot) {
                               POLYBENCH_GEMVER_BODY3_RAJA;
                              };
  auto poly_gemver_lam4 = [=] (Index_type i, 
                               Real_type &dot) {
                               POLYBENCH_GEMVER_BODY4_RAJA;
                              };
  auto poly_gemver_lam5 = [=] (Index_type i) {
                               POLYBENCH_GEMVER_BODY5_RAJA;
                              };
  auto poly_gemver_lam6 = [=] (Index_type i, 
                               Real_type &dot) {
                               POLYBENCH_GEMVER_BODY6_RAJA;
                              };
  auto poly_gemver_lam7 = [=] (Index_type i, Index_type j, Real_type &dot) {
                               POLYBENCH_GEMVER_BODY7_RAJA;
                              };
  auto poly_gemver_lam8 = [=] (Index_type i, 
                               Real_type &dot) {
                               POLYBENCH_GEMVER_BODY8_RAJA;
                              };

#else

  auto poly_gemver_lam1 = [=] (Index_type i, Index_type j) {
                               POLYBENCH_GEMVER_BODY1_RAJA;
                              };
  auto poly_gemver_lam2 = [=] (Index_type /* i */, Index_type /* j */,
                               Real_type &dot) {
                               POLYBENCH_GEMVER_BODY2_RAJA;
                              };
  auto poly_gemver_lam3 = [=] (Index_type i, Index_type j, Real_type &dot) {
                               POLYBENCH_GEMVER_BODY3_RAJA;
                              };
  auto poly_gemver_lam4 = [=] (Index_type i, Index_type /* j */,
                               Real_type &dot) {
                               POLYBENCH_GEMVER_BODY4_RAJA;
                              };
  auto poly_gemver_lam5 = [=] (Index_type i) {
                               POLYBENCH_GEMVER_BODY5_RAJA;
                              };
  auto poly_gemver_lam6 = [=] (Index_type i, Index_type /* j */,
                               Real_type &dot) {
                               POLYBENCH_GEMVER_BODY6_RAJA;
                              };
  auto poly_gemver_lam7 = [=] (Index_type i, Index_type j, Real_type &dot) {
                               POLYBENCH_GEMVER_BODY7_RAJA;
                              };
  auto poly_gemver_lam8 = [=] (Index_type i, Index_type /* j */,
                               Real_type &dot) {
                               POLYBENCH_GEMVER_BODY8_RAJA;
                              };
#endif

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMVER_BODY1;
          }
        }

        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY2;
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMVER_BODY3;
          }
          POLYBENCH_GEMVER_BODY4;
        } 

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY5;
        }

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY6;
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMVER_BODY7;
          }
          POLYBENCH_GEMVER_BODY8;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            poly_gemver_base_lam1(i, j);
          }
        }

        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY2;
          for (Index_type j = 0; j < n; j++) {
            poly_gemver_base_lam3(i, j, dot);
          }
          poly_gemver_base_lam4(i, dot);
        }

        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          poly_gemver_base_lam5(i);
        }

        #pragma omp parallel for
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMVER_BODY6;
          for (Index_type j = 0; j < n; j++) {
            poly_gemver_base_lam7(i, j, dot);
          }
          poly_gemver_base_lam8(i, dot);
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ_ARGS) || defined(RUN_RAJA_SEQ_ARGS_DEV)

    case RAJA_OpenMP : {

      using EXEC_POL1 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
#ifdef RUN_RAJA_SEQ_ARGS_DEV
              RAJA::statement::Lambda<0, RAJA::Segs<0,1>>
#else
              RAJA::statement::Lambda<0, RAJA::statement::Segs<0,1>>
#endif
            >
          >
        >;

      using EXEC_POL24 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
#ifdef RUN_RAJA_SEQ_ARGS_DEV
            RAJA::statement::Lambda<0, RAJA::Params<0>>,                               
#else
            RAJA::statement::Lambda<0, RAJA::statement::Params<0>>,                               
#endif
            RAJA::statement::For<1, RAJA::loop_exec,
#ifdef RUN_RAJA_SEQ_ARGS_DEV
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
#else
              RAJA::statement::Lambda<1, RAJA::statement::Segs<0,1>, RAJA::statement::Params<0>>
#endif
            >,
#ifdef RUN_RAJA_SEQ_ARGS_DEV
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
#else
            RAJA::statement::Lambda<2, RAJA::statement::Segs<0>, RAJA::statement::Params<0>>
#endif
          >
        >;

      using EXEC_POL3 = RAJA::loop_exec;

      using EXEC_POL5 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
#ifdef RUN_RAJA_SEQ_ARGS_DEV
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
#else
            RAJA::statement::Lambda<0, RAJA::statement::Segs<0>, RAJA::statement::Params<0>>,
#endif
            RAJA::statement::For<1, RAJA::loop_exec,
#ifdef RUN_RAJA_SEQ_ARGS_DEV
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
#else
              RAJA::statement::Lambda<1, RAJA::statement::Segs<0,1>, RAJA::statement::Params<0>>
#endif
            >,
#ifdef RUN_RAJA_SEQ_ARGS_DEV
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
#else
            RAJA::statement::Lambda<2, RAJA::statement::Segs<0>, RAJA::statement::Params<0>>
#endif
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment(0, n),
                                                  RAJA::RangeSegment(0, n)),
          poly_gemver_lam1
        );

        RAJA::kernel_param<EXEC_POL24>(
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
          RAJA::tuple<Real_type> {0.0},

          poly_gemver_lam2,
          poly_gemver_lam3,
          poly_gemver_lam4
        );

        RAJA::forall<EXEC_POL3> (RAJA::RangeSegment{0, n},
          poly_gemver_lam5
        );

        RAJA::kernel_param<EXEC_POL5>(
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
          RAJA::tuple<Real_type> {0.0},

          poly_gemver_lam6,
          poly_gemver_lam7,
          poly_gemver_lam8

        ); 

      }
      stopTimer();

      break;
    }

#else

    case RAJA_OpenMP : {

      using EXEC_POL1 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      using EXEC_POL24 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>
          >
        >;

      using EXEC_POL3 = RAJA::loop_exec;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                  RAJA::RangeSegment{0, n}),
          poly_gemver_lam1
        );

        RAJA::kernel_param<EXEC_POL24>(
          RAJA::make_tuple(RAJA::RangeSegment{0, n},
                           RAJA::RangeSegment{0, n}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_gemver_lam2,
          poly_gemver_lam3,
          poly_gemver_lam4
        );

        RAJA::forall<EXEC_POL3> (RAJA::RangeSegment{0, n},
          poly_gemver_lam5
        );

        RAJA::kernel_param<EXEC_POL24>(
          RAJA::make_tuple(RAJA::RangeSegment{0, n},
                           RAJA::RangeSegment{0, n}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_gemver_lam6,
          poly_gemver_lam7,
          poly_gemver_lam8

        );
      }
      stopTimer();

      break;
    }

#endif

    default : {
      std::cout << "\n  POLYBENCH_GEMVER : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
