//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void GEN_LIN_RECUR::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  auto genlinrecur_lam1 = [=](Index_type k) {
                            GEN_LIN_RECUR_BODY1;
                          };
  auto genlinrecur_lam2 = [=](Index_type i) {
                            GEN_LIN_RECUR_BODY2;
                          };

  switch ( vid ) {

    case Base_StdPar : {

      auto rangeK = std::views::iota((Index_type)0,N);
      auto rangeI = std::views::iota((Index_type)1,N+1);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //for (Index_type k = 0; k < N; ++k ) {
        std::for_each( std::execution::par_unseq,
                        std::begin(rangeK), std::end(rangeK),
                        [=](Index_type k) {
          GEN_LIN_RECUR_BODY1;
        });

        //for (Index_type i = 1; i < N+1; ++i ) {
        std::for_each( std::execution::par_unseq,
                        std::begin(rangeI), std::end(rangeI),
                        [=](Index_type i) {
          GEN_LIN_RECUR_BODY2;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto rangeK = std::views::iota((Index_type)0,N);
      auto rangeI = std::views::iota((Index_type)1,N+1);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //for (Index_type k = 0; k < N; ++k ) {
        std::for_each( std::execution::par_unseq,
                        std::begin(rangeK), std::end(rangeK),
                        [=](Index_type k) {
          genlinrecur_lam1(k);
        });

        //for (Index_type i = 1; i < N+1; ++i ) {
        std::for_each( std::execution::par_unseq,
                        std::begin(rangeI), std::end(rangeI),
                        [=](Index_type i) {
          genlinrecur_lam2(i);
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(0, N), genlinrecur_lam1);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(1, N+1), genlinrecur_lam2);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
