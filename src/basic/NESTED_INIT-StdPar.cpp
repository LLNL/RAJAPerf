//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void NESTED_INIT::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

#ifdef USE_STDPAR_COLLAPSE
      auto begin = counting_iterator<Index_type>(0);
      auto end   = counting_iterator<Index_type>(ni*nj*nk);
#else
      auto begin = counting_iterator<Index_type>(0);
      auto end   = counting_iterator<Index_type>(nk);
#endif

  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
                          NESTED_INIT_BODY;
                        };

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type idx) {
              const auto k  = idx / (nj*ni);
              const auto ij = idx % (nj*ni);
              const auto j  = ij / ni;
              const auto i  = ij % ni;
#else
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type k) {
            for (Index_type j = 0; j < nj; ++j )
              for (Index_type i = 0; i < ni; ++i )
#endif
              {
                NESTED_INIT_BODY;
                //std::cout << i << "," << j << "," << k << ";" << idx << " PAR\n";
              }
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type idx) {
              const auto k  = idx / (nj*ni);
              const auto ij = idx % (nj*ni);
              const auto j  = ij / ni;
              const auto i  = ij % ni;
#else
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type k) {
            for (Index_type j = 0; j < nj; ++j )
              for (Index_type i = 0; i < ni; ++i )
#endif
              {
                nestedinit_lam(i, j, k);
              }
          });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      using EXEC_POL = 
        RAJA::KernelPolicy<
          RAJA::statement::For<2, RAJA::loop_exec,    // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::For<0, RAJA::loop_exec,// i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                                 RAJA::RangeSegment(0, nj),
                                                 RAJA::RangeSegment(0, nk)),
                                nestedinit_lam
                              );

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
