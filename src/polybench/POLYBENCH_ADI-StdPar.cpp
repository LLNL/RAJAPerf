//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ADI.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_ADI::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_ADI_DATA_SETUP;

  counting_iterator<Index_type> begin(1);
  counting_iterator<Index_type> end(n-1);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) { 

          std::for_each( std::execution::par_unseq,
                         begin, end,
                         [=](Index_type i) {
            POLYBENCH_ADI_BODY2;
            for (Index_type j = 1; j < n-1; ++j) {
              POLYBENCH_ADI_BODY3;
            }  
            POLYBENCH_ADI_BODY4;
            for (Index_type k = n-2; k >= 1; --k) {
              POLYBENCH_ADI_BODY5;
            }  
          });

          std::for_each( std::execution::par_unseq,
                         begin, end,
                         [=](Index_type i) {
            POLYBENCH_ADI_BODY6;
            for (Index_type j = 1; j < n-1; ++j) {
              POLYBENCH_ADI_BODY7;
            }
            POLYBENCH_ADI_BODY8;
            for (Index_type k = n-2; k >= 1; --k) {
              POLYBENCH_ADI_BODY9;
            }  
          });

        }  // tstep loop

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_adi_base_lam2 = [=](Index_type i) {
                                  POLYBENCH_ADI_BODY2;
                                };
      auto poly_adi_base_lam3 = [=](Index_type i, Index_type j) {
                                  POLYBENCH_ADI_BODY3;
                                };
      auto poly_adi_base_lam4 = [=](Index_type i) {
                                  POLYBENCH_ADI_BODY4;
                                };
      auto poly_adi_base_lam5 = [=](Index_type i, Index_type k) {
                                  POLYBENCH_ADI_BODY5;
                                };
      auto poly_adi_base_lam6 = [=](Index_type i) {
                                  POLYBENCH_ADI_BODY6;
                                };
      auto poly_adi_base_lam7 = [=](Index_type i, Index_type j) {
                                  POLYBENCH_ADI_BODY7;
                                };
      auto poly_adi_base_lam8 = [=](Index_type i) {
                                  POLYBENCH_ADI_BODY8;
                                };
      auto poly_adi_base_lam9 = [=](Index_type i, Index_type k) {
                                  POLYBENCH_ADI_BODY9;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) {

          std::for_each( std::execution::par_unseq,
                         begin, end,
                          [=](Index_type i) {
            poly_adi_base_lam2(i);
            for (Index_type j = 1; j < n-1; ++j) {
              poly_adi_base_lam3(i, j);
            }
            poly_adi_base_lam4(i);
            for (Index_type k = n-2; k >= 1; --k) {
              poly_adi_base_lam5(i, k);
            }
          });

          std::for_each( std::execution::par_unseq,
                         begin, end,
                          [=](Index_type i) {
            poly_adi_base_lam6(i);
            for (Index_type j = 1; j < n-1; ++j) {
              poly_adi_base_lam7(i, j);
            }
            poly_adi_base_lam8(i);
            for (Index_type k = n-2; k >= 1; --k) {
              poly_adi_base_lam9(i, k);
            }
          });

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      POLYBENCH_ADI_VIEWS_RAJA;

      auto poly_adi_lam2 = [=](Index_type i) {
                             POLYBENCH_ADI_BODY2_RAJA;
                           };
      auto poly_adi_lam3 = [=](Index_type i, Index_type j) {
                             POLYBENCH_ADI_BODY3_RAJA;
                           };
      auto poly_adi_lam4 = [=](Index_type i) {
                             POLYBENCH_ADI_BODY4_RAJA;
                           };
      auto poly_adi_lam5 = [=](Index_type i, Index_type k) {
                             POLYBENCH_ADI_BODY5_RAJA;
                           };
      auto poly_adi_lam6 = [=](Index_type i) {
                             POLYBENCH_ADI_BODY6_RAJA;
                           };
      auto poly_adi_lam7 = [=](Index_type i, Index_type j) {
                             POLYBENCH_ADI_BODY7_RAJA;
                           };
      auto poly_adi_lam8 = [=](Index_type i) {
                             POLYBENCH_ADI_BODY8_RAJA;
                           };
      auto poly_adi_lam9 = [=](Index_type i, Index_type k) {
                             POLYBENCH_ADI_BODY9_RAJA;
                           };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0, RAJA::Segs<0>>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<3, RAJA::Segs<0,2>>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) { 

          RAJA::kernel<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            poly_adi_lam2,
            poly_adi_lam3,
            poly_adi_lam4,
            poly_adi_lam5

          );

          RAJA::kernel<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            poly_adi_lam6,
            poly_adi_lam7,
            poly_adi_lam8,
            poly_adi_lam9

          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      getCout() << "\nPOLYBENCH_ADI  Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf