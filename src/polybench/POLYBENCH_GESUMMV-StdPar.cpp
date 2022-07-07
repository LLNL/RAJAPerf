//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_GESUMMV::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_GESUMMV_DATA_SETUP;

  counting_iterator<Index_type> begin(0);
  counting_iterator<Index_type> end(N);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin, end, [=](Index_type i) {
          POLYBENCH_GESUMMV_BODY1;
          std::for_each(begin, end, [=,&tmpdot,&ydot](Index_type j) {
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

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin, end, [=](Index_type i) {
          POLYBENCH_GESUMMV_BODY1;
          std::for_each(begin, end, [=,&tmpdot,&ydot](Index_type j) {
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
