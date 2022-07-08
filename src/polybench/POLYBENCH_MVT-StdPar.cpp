//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>


namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_MVT::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_MVT_DATA_SETUP;

  counting_iterator<Index_type> begin(0);
  counting_iterator<Index_type> end(N);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin, end,
                       [=](Index_type i) {
          POLYBENCH_MVT_BODY1;
          std::for_each( std::execution::unseq,
                         begin, end,
                         [=,&dot](Index_type j) {
            POLYBENCH_MVT_BODY2;
          });
          POLYBENCH_MVT_BODY3;
        });

        std::for_each( std::execution::par_unseq,
                       begin, end,
                        [=](Index_type i) {
          POLYBENCH_MVT_BODY4;
          std::for_each( std::execution::unseq,
                         begin, end,
                         [=,&dot](Index_type j) {
            POLYBENCH_MVT_BODY5;
          });
          POLYBENCH_MVT_BODY6;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin, end,
                        [=](Index_type i) {
          POLYBENCH_MVT_BODY1;
          std::for_each( std::execution::unseq,
                         begin, end,
                         [=,&dot](Index_type j) {
            poly_mvt_base_lam2(i, j, dot);
          });
          poly_mvt_base_lam3(i, dot);
        });

        std::for_each( std::execution::par_unseq,
                       begin, end,
                        [=](Index_type i) {
          POLYBENCH_MVT_BODY4;
          std::for_each( std::execution::unseq,
                         begin, end,
                         [=,&dot](Index_type j) {
            poly_mvt_base_lam5(i, j, dot);
          });
          poly_mvt_base_lam6(i, dot);
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

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
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RAJA::kernel_param<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::tuple<Real_type>{0.0},
 
            poly_mvt_lam1,
            poly_mvt_lam2,
            poly_mvt_lam3
 
          );

          RAJA::kernel_param<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::tuple<Real_type>{0.0},
 
            poly_mvt_lam4,
            poly_mvt_lam5, 
            poly_mvt_lam6
 
          );

        }); // end sequential region (for single-source code)

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      getCout() << "\n  POLYBENCH_MVT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
