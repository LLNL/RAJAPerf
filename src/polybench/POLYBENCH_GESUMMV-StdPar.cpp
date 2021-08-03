//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

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

void POLYBENCH_GESUMMV::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_GESUMMV_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota((Index_type)0, N);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          POLYBENCH_GESUMMV_BODY1;
          std::for_each( std::begin(range), std::end(range), [=,&tmpdot,&ydot](Index_type j) {
            POLYBENCH_GESUMMV_BODY2;
          });
          POLYBENCH_GESUMMV_BODY3;
        });

      }
      stopTimer();

      break;
    }


    case Lambda_StdPar : {

      auto poly_gesummv_base_lam2 = [=](Index_type i, Index_type j, 
                                        Real_type& tmpdot, Real_type& ydot) {
                                      POLYBENCH_GESUMMV_BODY2;
                                    };
      auto poly_gesummv_base_lam3 = [=](Index_type i,
                                        Real_type& tmpdot, Real_type& ydot) {
                                      POLYBENCH_GESUMMV_BODY3;
                                    };

      auto range = std::views::iota((Index_type)0, N);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       std::begin(range), std::end(range), [=](Index_type i) {
          POLYBENCH_GESUMMV_BODY1;
          std::for_each( std::begin(range), std::end(range), [=,&tmpdot,&ydot](Index_type j) {
            poly_gesummv_base_lam2(i, j, tmpdot, ydot);
          });
          poly_gesummv_base_lam3(i, tmpdot, ydot);
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      POLYBENCH_GESUMMV_VIEWS_RAJA;

      auto poly_gesummv_lam1 = [=](Real_type& tmpdot, Real_type& ydot) {
                                   POLYBENCH_GESUMMV_BODY1_RAJA;
                                  };
      auto poly_gesummv_lam2 = [=](Index_type i, Index_type j, 
                                   Real_type& tmpdot, Real_type& ydot) {
                                   POLYBENCH_GESUMMV_BODY2_RAJA;
                                  };
      auto poly_gesummv_lam3 = [=](Index_type i,
                                   Real_type& tmpdot, Real_type& ydot) {
                                   POLYBENCH_GESUMMV_BODY3_RAJA;
                                  };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0, RAJA::Params<0,1>>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Params<0,1>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0,1>>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple( RAJA::RangeSegment{0, N},
                            RAJA::RangeSegment{0, N} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0), 
                           static_cast<Real_type>(0.0)),

          poly_gesummv_lam1,
          poly_gesummv_lam2,
          poly_gesummv_lam3
        );

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  POLYBENCH_GESUMMV : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
