//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

#define USE_STDPAR_COLLAPSE 1

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_GEMVER::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMVER_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota((Index_type)0, n);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          std::for_each( std::begin(range), std::end(range), [=](Index_type j) {
            POLYBENCH_GEMVER_BODY1;
          });
        });

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          POLYBENCH_GEMVER_BODY2;
          std::for_each( std::begin(range), std::end(range), [=,&dot](Index_type j) {
            POLYBENCH_GEMVER_BODY3;
          });
          POLYBENCH_GEMVER_BODY4;
        });

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          POLYBENCH_GEMVER_BODY5;
        });

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          POLYBENCH_GEMVER_BODY6;
          std::for_each( std::begin(range), std::end(range), [=,&dot](Index_type j) {
            POLYBENCH_GEMVER_BODY7;
          });
          POLYBENCH_GEMVER_BODY8;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

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

      auto range = std::views::iota((Index_type)0, n);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          std::for_each( std::begin(range), std::end(range), [=](Index_type j) {
            poly_gemver_base_lam1(i, j);
          });
        });

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          POLYBENCH_GEMVER_BODY2;
          std::for_each( std::begin(range), std::end(range), [=,&dot](Index_type j) {
            poly_gemver_base_lam3(i, j, dot);
          });
          poly_gemver_base_lam4(i, dot);
        });

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          poly_gemver_base_lam5(i);
        });

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          POLYBENCH_GEMVER_BODY6;
          std::for_each( std::begin(range), std::end(range), [=,&dot](Index_type j) {
            poly_gemver_base_lam7(i, j, dot);
          });
          poly_gemver_base_lam8(i, dot);
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      POLYBENCH_GEMVER_VIEWS_RAJA;

      auto poly_gemver_lam1 = [=] (Index_type i, Index_type j) {
                                   POLYBENCH_GEMVER_BODY1_RAJA;
                                  };
      auto poly_gemver_lam2 = [=] (Real_type &dot) {
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
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0, RAJA::Segs<0,1>>
            >
          >
        >;

      using EXEC_POL2 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >;

      using EXEC_POL3 = RAJA::loop_exec;

      using EXEC_POL4 =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                  RAJA::RangeSegment{0, n}),
          poly_gemver_lam1
        );
        
        RAJA::kernel_param<EXEC_POL2>( 
          RAJA::make_tuple(RAJA::RangeSegment{0, n},
                           RAJA::RangeSegment{0, n}),
          RAJA::tuple<Real_type>{0.0},

          poly_gemver_lam2,
          poly_gemver_lam3,
          poly_gemver_lam4
        );
        
        RAJA::forall<EXEC_POL3> (RAJA::RangeSegment{0, n}, 
          poly_gemver_lam5
        );

        RAJA::kernel_param<EXEC_POL4>( 
          RAJA::make_tuple(RAJA::RangeSegment{0, n},
                           RAJA::RangeSegment{0, n}),
          RAJA::tuple<Real_type>{0.0},

          poly_gemver_lam6,
          poly_gemver_lam7,
          poly_gemver_lam8

        );
        
      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  POLYBENCH_GEMVER : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
